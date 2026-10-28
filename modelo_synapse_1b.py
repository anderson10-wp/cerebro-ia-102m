#!/usr/bin/env python3
"""
ARQUITETURA OFICIAL DO SYNAPSE 1B (MoE ~1.2B TOTAL / 261M ATIVO)
Desenvolvido por Nandinho.
Especificação Técnica:
- Dimensão: 768
- Camadas: 20
- Cabeças de Atenção: 12 (head_dim = 64)
- Especialistas: 12 Especialistas SwiGLU (hidden_dim = 2048)
- Roteamento: Top-2 por token
- Positional Embedding: RoPE (Rotary Position Embeddings)
- Atenção: Fused QKV com suporte a KV-Cache
- Vocabulário: 32.768 tokens (Tied weights com LM Head)
"""

import math
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Enable UTF-8 for console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Config1B:
    vocab_size: int = 32768
    dim: int = 768
    layers: int = 20
    n_heads: int = 12
    head_dim: int = 64  # 12 * 64 = 768
    hidden_dim: int = 2048  # ~8/3 * 768 = 2048
    num_experts: int = 12
    top_k: int = 2
    context: int = 1024
    norm_eps: float = 1e-5

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class Attention(nn.Module):
    def __init__(self, cfg: Config1B):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        
        inv = 10000.0 ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        angles = torch.outer(torch.arange(cfg.context).float(), inv)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def rotate(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # x: [b, n_heads, t, head_dim]
        length = x.size(-2)
        cos = self.cos[start_pos:start_pos+length].to(x.dtype)[None, None]
        sin = self.sin[start_pos:start_pos+length].to(x.dtype)[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, t, d = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        
        q = self.rotate(q, start_pos=start_pos)
        k = self.rotate(k, start_pos=start_pos)
        
        # Suporte a KV-Cache para inferência veloz na GPU
        if kv_cache is not None and kv_cache[0] is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        new_kv_cache = (k, v)
        
        is_causal = (t > 1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out), new_kv_cache

class MoESwiGLUExpert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

class MoELayer(nn.Module):
    def __init__(self, cfg: Config1B):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.dim, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([
            MoESwiGLUExpert(cfg.dim, cfg.hidden_dim) for _ in range(cfg.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        x_flat = x.view(-1, d)
        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)
        
        # Seleciona Top-2 especialistas
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Caminho Ultrarrápido de Inferência para 1 Token (Elimina 12 loops e syncs de GPU)
        if x_flat.shape[0] == 1:
            e1 = top_k_indices[0, 0].item()
            e2 = top_k_indices[0, 1].item()
            w1 = top_k_weights[0, 0].unsqueeze(-1)
            w2 = top_k_weights[0, 1].unsqueeze(-1)
            out = self.experts[e1](x_flat) * w1 + self.experts[e2](x_flat) * w2
            return out.view(b, t, d), 0.0

        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i)
            if mask.any():
                token_indices, k_pos = torch.where(mask)
                tokens_in = x_flat[token_indices]
                expert_out = expert(tokens_in)
                weights = top_k_weights[token_indices, k_pos].unsqueeze(-1)
                out[token_indices] = out[token_indices] + (expert_out * weights).to(out.dtype)
        
        # Load balancing auxiliary loss
        f = (top_k_indices.unsqueeze(-1) == torch.arange(self.num_experts, device=x.device)).float().sum(dim=[0, 1]) / (b * t * self.top_k)
        P = probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(f * P)
        
        return out.view(b, t, d), aux_loss

class MoEBlock(nn.Module):
    def __init__(self, cfg: Config1B):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.norm2 = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attn = Attention(cfg)
        self.moe = MoELayer(cfg)

    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv_cache = self.attn(self.norm1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + attn_out
        moe_out, aux_loss = self.moe(self.norm2(x))
        return x + moe_out, aux_loss, new_kv_cache

class BrainMoE1B(nn.Module):
    def __init__(self, cfg: Optional[Config1B] = None):
        super().__init__()
        self.cfg = cfg or Config1B()
        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.dim)
        self.blocks = nn.ModuleList([MoEBlock(self.cfg) for _ in range(self.cfg.layers)])
        self.norm = RMSNorm(self.cfg.dim, eps=self.cfg.norm_eps)
        self.head = nn.Linear(self.cfg.dim, self.cfg.vocab_size, bias=False)
        
        # Tied weights (LM Head compartilha pesos do embedding para economizar 25M de params)
        self.head.weight = self.embedding.weight

    def forward(
        self, 
        tokens: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0
    ):
        if tokens.shape[1] > self.cfg.context and kv_caches is None:
            raise ValueError(f"Sequência ({tokens.shape[1]}) excede contexto configurado ({self.cfg.context})")
            
        x = self.embedding(tokens)
        total_aux_loss = 0.0
        new_kv_caches = []
        
        for i, block in enumerate(self.blocks):
            layer_kv = kv_caches[i] if kv_caches is not None else None
            x, aux, n_kv = block(x, kv_cache=layer_kv, start_pos=start_pos)
            total_aux_loss = total_aux_loss + aux
            if kv_caches is not None:
                new_kv_caches.append(n_kv)
                
        logits = self.head(self.norm(x))
        
        if kv_caches is not None:
            return logits, new_kv_caches
            
        if targets is None:
            return logits, total_aux_loss
            
        loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size).float(), targets.reshape(-1), ignore_index=-100)
        return logits, loss, total_aux_loss

    def count_parameters(self) -> Tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        # Parâmetros ativos por token
        emb_params = self.embedding.weight.numel()
        attn_per_layer = sum(p.numel() for p in self.blocks[0].attn.parameters())
        expert_params = sum(p.numel() for p in self.blocks[0].moe.experts[0].parameters())
        norm_params = self.blocks[0].norm1.weight.numel() + self.blocks[0].norm2.weight.numel()
        router_params = self.blocks[0].moe.router.weight.numel()
        
        active_per_layer = attn_per_layer + (self.cfg.top_k * expert_params) + norm_params + router_params
        active_params = emb_params + (self.cfg.layers * active_per_layer) + self.norm.weight.numel()
        return total_params, active_params

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 AUDITORIA ARQUITETURAL: SYNAPSE 1B (MoE 12x Top-2)")
    print("=" * 70)
    cfg = Config1B()
    model = BrainMoE1B(cfg)
    total_p, active_p = model.count_parameters()
    
    print(f"• Dimensão Oculta (`dim`):           {cfg.dim}")
    print(f"• Número de Camadas (`layers`):       {cfg.layers}")
    print(f"• Cabeças de Atenção (`n_heads`):     {cfg.n_heads}")
    print(f"• Especialistas por Camada:          {cfg.num_experts} (SwiGLU)")
    print(f"• Especialistas Ativos por Token:    {cfg.top_k}")
    print(f"• Tamanho do Vocabulário:            {cfg.vocab_size:,}")
    print("-" * 70)
    print(f"📊 PARÂMETROS TOTAIS:                {total_p:,} (~{total_p/1e9:.2f} Bilhão)")
    print(f"⚡ PARÂMETROS ATIVOS (POR TOKEN):    {active_p:,} (~{active_p/1e6:.1f} Milhões)")
    print(f"💾 Consumo Estimado em VRAM (FP16):  ~{(total_p * 2) / (1024**3):.2f} GB")
    print("=" * 70)
    
    # Teste de Sanidade de Tensores
    print("🧪 Executando teste de Forward com batch dummy...")
    dummy_input = torch.randint(0, cfg.vocab_size, (1, 16))
    dummy_targets = torch.randint(0, cfg.vocab_size, (1, 16))
    logits, loss, aux = model(dummy_input, targets=dummy_targets)
    print(f"✅ Forward OK! Logits shape: {logits.shape} | Loss inicial: {loss.item():.4f} | Aux Loss: {aux.item():.4f}")
    
    # Teste de KV-Cache
    print("🧪 Testando KV-Cache de inferência sequencial...")
    dummy_token = torch.tensor([[42]])
    kv_caches = [(None, None)] * cfg.layers
    out_logits, kv_caches = model(dummy_token, kv_caches=kv_caches, start_pos=0)
    print(f"✅ KV-Cache OK! Próximo token inferido com shape: {out_logits.shape}")
    print("=" * 70)
    print("🎯 ARQUITETURA SYNAPSE 1B HOMOLOGADA E PRONTA PARA PRODUÇÃO!")
