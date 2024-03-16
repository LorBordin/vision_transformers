from torch import nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(TransformerBlock, self).__init__()

        # Attributes
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MultiHeadAttention(hidden_d, n_heads)
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


class MultiHeadAttention(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MultiHeadAttention, self).__init__()

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