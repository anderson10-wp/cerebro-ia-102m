"""Decoder language model initialized from scratch; no downloaded model weights."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 32768
    dim: int = 640
    layers: int = 16
    heads: int = 10
    hidden_dim: int = 1792
    context: int = 1024
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        z = x.float()
        return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.dim // cfg.heads
        assert cfg.dim % cfg.heads == 0 and self.head_dim % 2 == 0
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        inv = cfg.rope_base ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        angles = torch.outer(torch.arange(cfg.context).float(), inv)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def rotate(self, x):
        length = x.size(-2)
        cos = self.cos[:length].to(x.dtype)[None, None]
        sin = self.sin[:length].to(x.dtype)[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        z = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v, is_causal=True)
        return self.out(z.transpose(1, 2).contiguous().view(b, t, d))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        z = self.norm2(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class Brain(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.dim)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.layers)])
        self.norm = RMSNorm(self.cfg.dim)
        self.head = nn.Linear(self.cfg.dim, self.cfg.vocab_size, bias=False)
        self.head.weight = self.embedding.weight
        self.apply(self._init)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=0.02 / math.sqrt(2 * self.cfg.layers))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * self.cfg.layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, targets=None):
        if tokens.shape[1] > self.cfg.context:
            raise ValueError('Sequence exceeds configured context')
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size).float(), targets.reshape(-1))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, tokens, new_tokens=100, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(new_tokens):
            logits = self(tokens[:, -self.cfg.context:])[:, -1].float() / max(temperature, 1e-5)
            cutoff = torch.topk(logits, min(top_k, logits.shape[-1]))[0][:, -1:]
            logits = logits.masked_fill(logits < cutoff, float('-inf'))
            token = torch.multinomial(logits.softmax(-1), 1)
            tokens = torch.cat((tokens, token), dim=1)
        return tokens
