"""Reference inference adapter for the trained numeric transformer checkpoints."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(width)
        self.ln2 = nn.LayerNorm(width)
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        self.fc = nn.Linear(width, 4 * width, bias=False)
        self.proj = nn.Linear(4 * width, width, bias=False)

    def forward(self, x):
        batch, length, width = x.shape
        h = self.ln1(x)
        q, k, v = [module(h).reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for module in (self.q, self.k, self.v)]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        x = x + self.o(out.transpose(1, 2).reshape(batch, length, width))
        x = x + self.proj(F.relu(self.fc(self.ln2(x))))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, width=128, heads=4, layers=4, vocab=32, context=96):
        super().__init__()
        self.tok = nn.Embedding(vocab, width)
        self.pos = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.final = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.normal_(block.o.weight, std=0.02 / (2 * layers) ** 0.5)
            nn.init.normal_(block.proj.weight, std=0.02 / (2 * layers) ** 0.5)

    def forward(self, ids):
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.output(self.final(x))
