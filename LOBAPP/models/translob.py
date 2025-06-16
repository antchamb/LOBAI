import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath


class TransLOB(nn.Module):
    """One-stop implementation of TransLOB model (Wallbridge, 2020)."""
    def __init__(
        self,
        *,
        seq_len: int          = 100,    # T
        n_features: int       = 40,
        cnn_out: int          = 14,
        transformer_dim: int  = 15,    # 15 = 14 (CNN out) + 1 (temporal encoding)
        depth: int            = 2,    # Number of transformer blocks
        heads: int            = 3,    # Number of attention heads
        mlp_ratio: int        = 4,    # expansion du FFN (MLP)
        num_classes: int      = 3,    # up, flat, down
        dropout: float        = 0.1,
        drop_path_rate: float = 0.0,
        debug: bool = False,
    ):
        super().__init__()
        self.seq_len, self.debug, self.heads = seq_len, debug, heads
        self.scale = (transformer_dim // heads) ** -0.5          # √d_k^-1

        # Construction CNN dilaté causal : padding gauche seul
        dilations = [2 ** i for i in range(5)]     # 5 dilations
        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(n_features if i == 0 else cnn_out,
                      cnn_out, kernel_size=2,
                      padding=0, dilation=d)  # FIX (pad=0)
            for i, d in enumerate(dilations)
        ])
        self.dilations = dilations  # FIX (mémorisées)
        self.cnn_act = nn.ReLU(inplace=True)

        # Normalisation + encodage temporel
        self.norm_in = nn.LayerNorm(cnn_out)
        self.temp_enc = nn.Parameter(torch.zeros(1, seq_len, 1))
        trunc_normal_(self.temp_enc, std=0.02)

        # Bloc Transformer partagé
        blk = nn.ModuleDict({
            "norm1": nn.LayerNorm(transformer_dim),
            "qkv": nn.Linear(transformer_dim, transformer_dim * 3),
            "proj": nn.Linear(transformer_dim, transformer_dim),
            "norm2": nn.LayerNorm(transformer_dim),
            "mlp": nn.Sequential(
                nn.Linear(transformer_dim, transformer_dim * mlp_ratio),
                nn.ReLU(inplace=True),  # FIX (ReLU)
                nn.Linear(transformer_dim * mlp_ratio, transformer_dim),
            ),
            "drop_path": DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity(),
        })
        self.block = blk  # FIX (poids partagés)
        self.depth = depth

        # Classifieur
        self.classifier = nn.Sequential(
            nn.Linear(seq_len * transformer_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    # ----------------- Sous-routine attention causale ------------------
    def _attn(self, x):
        B, N, C = x.shape
        qkv = (self.block["qkv"](x)
               .reshape(B, N, 3, self.heads, C // self.heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)  # FIX (dropout retiré)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.block["proj"](x)  # FIX (pas de proj_drop)

    # --------------------------- Forward ------------------------------
    def forward(self, x):
        B, T, C = x.shape
        if T != self.seq_len or C != 40:
            raise ValueError("Attendu (B,100,40)")

        # 1. CNN causal
        x = x.transpose(1, 2)  # (B,40,T)
        for conv, d in zip(self.cnn_layers, self.dilations):
            x = F.pad(x, (d, 0))  # FIX (pad gauche)
            x = self.cnn_act(conv(x))
        x = x.transpose(1, 2)  # (B,100,14)

        # 2. LN + canal temporel
        x = self.norm_in(x)
        x = torch.cat((x, self.temp_enc.repeat(B, 1, 1)), dim=-1)  # (B,100,15)

        # 3. Transformer : même bloc appliqué ‘depth’ fois
        for _ in range(self.depth):
            y = self._attn(self.block["norm1"](x))
            x = x + self.block["drop_path"](y)
            x = x + self.block["drop_path"](self.block["mlp"](self.block["norm2"](x)))

        # 4. Classification
        logits = self.classifier(x.flatten(1))
        return F.softmax(logits, dim=-1)

torch.manual_seed(0)
model = TransLOB()
dummy = torch.randn(8, 100, 40)
out = model(dummy)
print(out.shape)  # torch.Size([8, 3])
print(out.sum(1))  # ≈ tensor([... 1.0000 ...])
