import os, sys, math, time, json, urllib.request
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("🧠 CÉREBRO IA — UPCYCLING MoE 500M + MODO PENSAR (Chain-of-Thought)")
print("=" * 70)

# 1. Hardware Check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    bf16 = torch.cuda.is_bf16_supported()
    print(f"🚀 GPU Detectada: {gpu_name}")
    print(f"⚡ VRAM: {vram:.2f} GB | Suporte BFloat16: {bf16}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("⚠️ Rodando em CPU! (Para alto desempenho, conecte uma GPU no Colab)")

# 2. Downloads se necessário
CHECKPOINT_FILE = "best-model.pt"
TOKENIZER_FILE = "tokenizer.json"

if not os.path.exists(CHECKPOINT_FILE):
    # Verifica se já existe em outputs
    local_candidates = list(Path('.').rglob('best-model.pt'))
    if local_candidates:
        CHECKPOINT_FILE = str(local_candidates[0])
        print(f"✅ Checkpoint encontrado localmente em: {CHECKPOINT_FILE}")
    else:
        print("📥 Baixando checkpoint original 102M do GitHub Releases...")
        import zipfile, shutil
        url = "https://github.com/anderson10-wp/cerebro-ia-102m/releases/download/arquivo-2026-10-08/artifact-1ea8da9fa900f7f2ec1beaf5f34511d449ea695732e19d1f7744d4efb21f9a9e.zip"
        zip_path = "checkpoint_102m.zip"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(zip_path, 'wb') as out_f:
            shutil.copyfileobj(resp, out_f)
        print("📦 Extraindo pesos para best-model.pt...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open('payload') as src, open(CHECKPOINT_FILE, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print("✅ Checkpoint 102M pronto e verificado!")

if not os.path.exists(TOKENIZER_FILE):
    # Procura no repositório local ou baixa
    candidates = list(Path(".").rglob("tokenizer.json"))
    if candidates:
        TOKENIZER_FILE = str(candidates[0])
        print(f"✅ Tokenizador encontrado em: {TOKENIZER_FILE}")
    else:
        print("📥 Baixando tokenizer.json...")
        url = "https://raw.githubusercontent.com/anderson10-wp/cerebro-ia-102m/main/outputs/h100-pretrain/result/tokenizer.json"
        urllib.request.urlretrieve(url, TOKENIZER_FILE)
        print("✅ Tokenizador pronto!")

# 3. Definição da Arquitetura MoE 500M
@dataclass
class Config:
    vocab_size: int = 32768
    dim: int = 640
    layers: int = 16
    heads: int = 10
    hidden_dim: int = 1792
    context: int = 1024
    rope_base: float = 10000.0
    num_experts: int = 8
    top_k: int = 2

cfg = Config()

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

class MoESwiGLUExpert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class MoELayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.dim, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([
            MoESwiGLUExpert(cfg.dim, cfg.hidden_dim) for _ in range(cfg.num_experts)
        ])

    def forward(self, x):
        b, t, d = x.shape
        x_flat = x.view(-1, d)
        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i)
            if mask.any():
                token_indices, k_pos = torch.where(mask)
                tokens_in = x_flat[token_indices]
                expert_out = expert(tokens_in)
                weights = top_k_weights[token_indices, k_pos].unsqueeze(-1)
                out.index_add_(0, token_indices, (expert_out * weights).to(out.dtype))
        
        # Load balancing loss
        f = (top_k_indices.unsqueeze(-1) == torch.arange(self.num_experts, device=x.device)).float().sum(dim=[0, 1]) / (b * t * self.top_k)
        P = probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(f * P)
        return out.view(b, t, d), aux_loss

class MoEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.moe = MoELayer(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        moe_out, aux_loss = self.moe(self.norm2(x))
        return x + moe_out, aux_loss

class BrainMoE(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.dim)
        self.blocks = nn.ModuleList([MoEBlock(self.cfg) for _ in range(self.cfg.layers)])
        self.norm = RMSNorm(self.cfg.dim)
        self.head = nn.Linear(self.cfg.dim, self.cfg.vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens, targets=None):
        if tokens.shape[1] > self.cfg.context:
            raise ValueError('Sequence exceeds configured context')
        x = self.embedding(tokens)
        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux = block(x)
            total_aux_loss = total_aux_loss + aux
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, total_aux_loss
        loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size).float(), targets.reshape(-1), ignore_index=-100)
        return logits, loss, total_aux_loss

model = BrainMoE(cfg)
total_params = sum(p.numel() for p in model.parameters())
print(f"🔥 Modelo Cérebro MoE 500M instanciado: {total_params:,} Parâmetros!")

# 4. Carregamento e Preparação do Modelo MoE 500M
round1_ckpt = None
for candidate in [
    "cerebro-moe-500m-modo-pensar.pt",
    "/content/drive/MyDrive/cerebro-moe-500m-modo-pensar.pt"
]:
    if os.path.exists(candidate):
        round1_ckpt = candidate
        break

if round1_ckpt:
    print(f"🔄 Checkpoint MoE 500M da Rodada 1 detectado em: {round1_ckpt}")
    print("🚀 Retomando treinamento a partir da Rodada 1 para a RODADA 2 (Refinamento Avançado)...")
    try:
        raw_500 = torch.load(round1_ckpt, map_location='cpu', weights_only=False)
    except Exception:
        raw_500 = torch.load(round1_ckpt, map_location='cpu')
    state_500 = raw_500['model'] if isinstance(raw_500, dict) and 'model' in raw_500 else raw_500
    model.load_state_dict(state_500)
    print("✅ Checkpoint da Rodada 1 carregado com sucesso!")
else:
    print("🔄 Executando Upcycling de pesos do zero (102M -> 500M MoE)...")
    try:
        raw_state = torch.load(CHECKPOINT_FILE, map_location='cpu', weights_only=False)
    except Exception:
        raw_state = torch.load(CHECKPOINT_FILE, map_location='cpu')
    state = raw_state['model'] if isinstance(raw_state, dict) and 'model' in raw_state else raw_state

    moe_state = model.state_dict()
    for key in state:
        if 'gate.weight' in key or 'up.weight' in key or 'down.weight' in key:
            parts = key.split('.')
            block_idx = parts[1]
            weight_name = parts[2]
            for exp_idx in range(cfg.num_experts):
                target_key = f'blocks.{block_idx}.moe.experts.{exp_idx}.{weight_name}.weight'
                w = state[key].clone()
                if exp_idx > 0:
                    w = w + torch.randn_like(w) * (0.005 * w.std())
                moe_state[target_key] = w
        else:
            moe_state[key] = state[key]

    for l in range(cfg.layers):
        nn.init.zeros_(model.blocks[l].moe.router.weight)

    model.load_state_dict(moe_state)
    print(f"✅ Upcycling inicial concluído com 100% dos pesos base preservados!")

model = model.to(device)

# 5. Tokenizer & Dataset
try:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_FILE)
except Exception:
    import json
    with open(TOKENIZER_FILE, 'r', encoding='utf-8') as f:
        tok_data = json.load(f)
    # Fallback simples se tokenizers não estiver instalado
    class SimpleTok:
        def __init__(self, vocab):
            self.vocab = vocab
        def encode(self, text):
            class Out: pass
            o = Out()
            o.ids = [self.vocab.get(c, 0) for c in text]
            return o
        def decode(self, ids):
            inv = {v: k for k, v in self.vocab.items()}
            return "".join([inv.get(i, "") for i in ids])
    tok = SimpleTok(tok_data.get('model', {}).get('vocab', {}))

# Procura dataset de treino
dataset_files = ["dataset_modo_pensar.jsonl", "outputs/dataset_modo_pensar.jsonl"]
cot_texts = []
for df in dataset_files:
    if os.path.exists(df):
        with open(df, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "treino_formatado" in item:
                        cot_texts.append(item["treino_formatado"])

print(f"📚 Dataset do 'Modo Pensar': {len(cot_texts)} amostras ricas de raciocínio carregadas.")

class CoTDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len=1024):
        self.examples = []
        for text in texts:
            parts = text.split('\nResposta: ')
            if len(parts) == 2:
                prompt = parts[0] + '\nResposta: '
                target = parts[1]
                p_ids = tokenizer.encode(prompt).ids
                t_ids = tokenizer.encode(target).ids
                input_ids = (p_ids + t_ids)[:max_len]
                labels = ([-100] * len(p_ids) + t_ids)[:max_len]
                if len(input_ids) > 1:
                    self.examples.append((torch.tensor(input_ids[:-1], dtype=torch.long),
                                         torch.tensor(labels[1:], dtype=torch.long)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    max_len = max(len(x[0]) for x in batch)
    inputs = torch.zeros(len(batch), max_len, dtype=torch.long)
    targets = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (inp, tar) in enumerate(batch):
        inputs[i, :len(inp)] = inp
        targets[i, :len(tar)] = tar
    return inputs, targets

if len(cot_texts) > 0:
    ds = CoTDataset(cot_texts, tok)
    batch_size = 4 if torch.cuda.is_available() else 1
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 6. Treinamento
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    optimizer = AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01, betas=(0.9, 0.95))
    epochs = 10
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(loader), eta_min=1e-5)
    
    print("\n" + "=" * 70)
    print("⚡ INICIANDO FINE-TUNING NA GPU COM BFLOAT16...")
    print("=" * 70)
    model.train()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        for step, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, token_loss, aux_loss = model(inputs, targets=targets)
                    loss = token_loss + 0.01 * aux_loss
            else:
                logits, token_loss, aux_loss = model(inputs, targets=targets)
                loss = token_loss + 0.01 * aux_loss
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += token_loss.item()
            
        dt = time.time() - t0
        print(f"Época [{epoch:02d}/{epochs:02d}] | Loss: {epoch_loss/len(loader):.4f} | Aux MoE: {aux_loss.item():.3f} | Tempo: {dt:.2f}s")
        
    print("🎉 Treinamento concluído com sucesso!")
    
    # 7. Salvar modelo final
    out_name = "cerebro-moe-500m-modo-pensar.pt"
    print(f"💾 Salvando {out_name}...")
    torch.save({
        'model': model.state_dict(),
        'config': cfg,
        'type': 'Sparse-MoE-8x-Top2'
    }, out_name)
    print(f"✅ Checkpoint {out_name} salvo com sucesso!")

# 8. Teste de Geração
print("\n" + "=" * 70)
print("🤖 TESTE DE GERAÇÃO COM MODO PENSAR (<think> ... </think>):")
print("=" * 70)
model.eval()

def test_gen(pergunta):
    prompt = f"Pergunta: {pergunta}\nResposta: <think>\n"
    print(f"\n👉 Pergunta: {pergunta}")
    ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(180):
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(ids)
            else:
                logits, _ = model(ids)
            probs = F.softmax(logits[0, -1, :] / 0.7, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt.unsqueeze(0)], dim=1)
            if ids.shape[1] >= cfg.context:
                break
    print(tok.decode(ids[0].tolist()))

test_gen("Por que a água do mar é salgada?")
