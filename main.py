from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import sys
import torch

np.random.seed(0)
torch.manual_seed(0)


class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()

        # Attributes
        self.chw = chw  # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1. Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2. Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3. Positional embedding
        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(n_patches**2 + 1, hidden_d),
            persistent=False,
        )

        # 4. Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        # 5. Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d), 
            nn.Softmax(dim=-1)
            )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]
        out = self.mlp(out)

        return out  
    
   
class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        
        self.d_head = d // n_heads
        self.n_heads = n_heads

        self.q_mapping = nn.Linear(d, d)
        self.k_mapping = nn.Linear(d, d)
        self.v_mapping = nn.Linear(d, d)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        q = self.q_mapping(sequences)
        k = self.k_mapping(sequences)
        v = self.v_mapping(sequences)

        q = q.view(q.size(0), q.size(1), self.n_heads, self.d_head)
        k = k.view(k.size(0), k.size(1), self.n_heads, self.d_head)
        v = v.view(v.size(0), v.size(1), self.n_heads, self.d_head)

        q = q.transpose(1, 2)  # (N, n_heads, seq_length, d_head)
        k = k.transpose(1, 2)  # (N, n_heads, seq_length, d_head)
        v = v.transpose(1, 2)  # (N, n_heads, seq_length, d_head)

        attention = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)
        attention = self.softmax(attention)

        out = torch.matmul(attention, v)  # (N, n_heads, seq_length, d_head)

        out = out.transpose(1, 2).contiguous().view(out.size(0), out.size(2), -1)  # (N, seq_length, d)

        return out

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()

        # Attributes
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"
    assert h % n_patches == 0, "Image size must be divisible by n_patches"

    patch_size = h // n_patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(n, n_patches * n_patches, c, patch_size, patch_size)
    patches = patches.permute(0, 1, 3, 4, 2).contiguous().view(n, n_patches * n_patches, -1)

    return patches


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d))) if j % 2 == 0 \
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result
            

def main(model, device, args):
    # Loading data
    transform = ToTensor()

    dataset_name = args["dataset"]
    n_epochs = args["n_epochs"]

    if dataset_name == "minst":
        from torchvision.datasets import MNIST as dataset
    elif dataset_name == "cifar10":
        from torchvision.datasets import CIFAR10 as dataset
    else:
        print("[ERROR] Unkown dataset.")
        sys.exit(0)
    
    train_set = dataset(
        root="./../datasets", train=True, download=True, transform=transform
    )
    test_set = dataset(
        root="./../datasets", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and training options
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
    )

    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(n_epochs, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
        ):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{n_epochs} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")

def print_model_stats(model, args):

    n_params = sum(p.numel() for p in model.parameters())

    print()
    print("*** ----- Model Stats ----- ***")
    print(f"[INFO] Input dimension: {args['input_size']}.")
    print(f"[INFO] Number of splits per size: {args['n_patches']}.")
    print(f"[INFO] Number of transformer blocks: {args['n_blocks']}.")
    print(f"[INFO] Embbedding dimension: {args['emb_size']}.")
    print(f"[INFO] Number of attention heads per block: {args['n_heads']}.")
    print(f"[INFO] Number of classification classes: {args['out_dim']}.")    
    print("*** _______________________ ***")
    print("[INFO] Number of training parameters:", n_params, end="\n")
    print("*** ----------------------- ***")
    print()


if __name__ == "__main__":

    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="minst",
        help="Dataset name. Options: 'mnist' (default), 'cifar10'.")
    ap.add_argument("-i", "--input-size", default="(1, 28, 28)", 
        help="Image input shape. Defaults to (1, 28, 28).")
    ap.add_argument("-p", "--n-patches", type=int, default=7, 
        help="Number of splits per side. Defaults to 7.")
    ap.add_argument("-b", "--n-blocks", type=int, default=2,
        help="Number of transformer blocks. Defaults to 2.")
    ap.add_argument("-e", "--emb-size", type=int, default=8, 
        help="Dimension of the embeddings. Defaults to 8.")
    ap.add_argument("-nh", "--n-heads", type=int, default=2,
        help="Number of Self Attention Heads per block. Defaults to 2.")
    ap.add_argument("-o", "--out-dim", type=int, default=10, 
        help="Number of prediction classes. Defaults to 10.")
    ap.add_argument('--n-epochs', type=int, default=5,
        help="Number of training epochs")
    args = vars(ap.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyViT(
        chw=eval(args["input_size"]), 
        n_patches=args["n_patches"], 
        n_blocks=args["n_blocks"], 
        hidden_d=args["emb_size"], 
        n_heads=args["n_heads"], 
        out_d=args["out_dim"]
    ).to(device)

    print_model_stats(model, args)
    main(model, device, args)
