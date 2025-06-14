import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath


def _assert_shape(tensor, expected_tail, name):
    """Utility to chech tensor shape (excluding batch dimension)."""
    if tuple(tensor.shape[1:]) != expected_tail:
        raise RuntimeError(
            f"{name}: expected shape (_, {', '.join(map(str, expected_tail))}), but got {tuple(tensor.shape)} "
        )

class TransLOB(nn.Module):
    """One-stop implementation of TransLOB model (Wallbridge, 2020)."""
    def __init__(
        self,
        *,
        seq_len: int = 100, # T
        n_features: int = 40,
        cnn_out: int = 14,
        transformer_dim: int = 15, # 15 = 14 (CNN out) + 1 (temporal encoding)
        depth: int = 2, # Number of transformer blocks
        heads: int = 3, # Number of attention heads
        mlp_ratio: int = 4, # expansion du FFN (MLP)
        num_classes: int = 3, # up, flat, down
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        debug: bool = False,
    ):
        super().__init__()
        self.debug = debug
        self.seq_len = seq_len
        kernel_size = 2 # fixed by the article
        pad = kernel_size - 1 # padding for causal convolution, ensure strict causality
        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=n_features if i == 0 else cnn_out,
                out_channels=cnn_out,
                kernel_size=kernel_size,
                padding=pad * (2 ** i), # dilation 2**i
                dilation=2 ** i
            )
            for i in range(5)
        ])
        self.cnn_act = nn.ReLU(inplace=True)
        self.norm_in = nn.LayerNorm(cnn_out)
        self.temp_enc = nn.Parameter(torch.zeros(1, seq_len, 1))
        trunc_normal_(self.temp_enc, std=0.02) # init gaussian truncated to 2 std

        # Transformer blocks
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(transformer_dim),
                "qkv": nn.Linear(transformer_dim, transformer_dim * 3, bias=True),
                "attn_drop": nn.Dropout(dropout),
                "proj": nn.Linear(transformer_dim, transformer_dim),
                "proj_drop": nn.Dropout(dropout),
                "drop_path": DropPath(dpr[i]) if dpr[i] > 0 else nn.Identity(),
                "norm2": nn.LayerNorm(transformer_dim),
                "mlp": nn.Sequential(
                    nn.Linear(transformer_dim, transformer_dim * mlp_ratio, bias=True),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(transformer_dim * mlp_ratio, transformer_dim),
                    nn.Dropout(dropout)
                )
            }))

        self.classifier = nn.Sequential(
            nn.Linear(seq_len * transformer_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        self.heads = heads
        self.scale = (transformer_dim // heads) ** -0.5 # np.sqrt(d_k) inverse

    def _causal_self_attention(self, x: torch.Tensor, block: nn.ModuleDict):
        B, N, C = x.shape
        qkv = block["qkv"](x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        mask = torch.tril(torch.ones(N, N, device = x.device, dtype=torch.bool))
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = block["attn_drop"](F.softmax(attn, dim=-1))

        x = attn @ v # (B, heads, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = block["proj_drop"](block["proj"](x))
        return x

    def forward(self, x: torch.Tensor):
        """
        Input shape: (batch_size, T, 40 (nb_features))
        Output: probabilities (batch, 3)
        """
        B, T, F = x.shape

        x_cnn = x.transpose(1, 2)
        for conv in self.cnn_layers:
            x_cnn = self.cnn_act(conv(x_cnn))
        x_cnn = x_cnn.transpose(1, 2)

        x = self.norm_in(x_cnn)
        x = torch.cat((x, self.temp_enc.repeat(B, 1, 1)), dim=-1)

        for blk in self.blocks:
            y = self._causal_self_attention(x, blk)
            x += blk["drop_path"](y)
            x += blk["drop_path"](blk["mlp"](blk["norm2"](x)))

        logits = self.classifier(x.flatten(start_dim=1))
        return F.softmax(logits, dim=-1)



torch.manual_seed(0)
model = TransLOB()
dummy = torch.randn(8, 100, 40)
out = model(dummy)
print(out.shape, out.sum(dim=1))