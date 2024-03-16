from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import sys
import torch

from vision_transformer.utils import print_model_stats
from vision_transformer import ViTModel

np.random.seed(0)
torch.manual_seed(0)
            

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
    model = ViTModel(
        chw=eval(args["input_size"]), 
        n_patches=args["n_patches"], 
        n_blocks=args["n_blocks"], 
        hidden_d=args["emb_size"], 
        n_heads=args["n_heads"], 
        out_d=args["out_dim"]
    ).to(device)

    print_model_stats(model, args)
    main(model, device, args)
