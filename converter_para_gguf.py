#!/usr/bin/env python3
"""
CONVERSOR NATIVO SYNAPSE 1.2B MOE -> FORMATO GGUF (PARA OLLAMA E LM STUDIO)
Projeto Synapse — Desenvolvido por Nandinho.

Converte o checkpoint PyTorch (.pt) de 4.8 GB para um arquivo .gguf em FP16 (~2.4 GB)
compatível com llama.cpp, Ollama e LM Studio, utilizando a arquitetura Qwen2MoE.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import gguf

# Suporte UTF-8 no Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
DOWNLOADS = Path(os.path.expanduser(r"~\Downloads"))

# Localiza o arquivo .pt
CANDIDATOS = [
    DOWNLOADS / "synapse-1b-r1-final.pt",
    WORKSPACE / "synapse-1b-r1-final.pt",
    DOWNLOADS / "synapse-1b-latest.pt",
    WORKSPACE / "synapse-1b-latest.pt"
]

CKPT_FILE = None
for c in CANDIDATOS:
    if c.exists():
        CKPT_FILE = c
        break

if not CKPT_FILE:
    print("❌ Checkpoint synapse-1b-r1-final.pt não encontrado!")
    sys.exit(1)

OUTPUT_GGUF = WORKSPACE / "synapse-1b-r1-fp16.gguf"
TOKENIZER_FILE = WORKSPACE / "tokenizer.json"

print("=" * 72)
print("📦 CONVERSOR GGUF SYNAPSE 1.2B MoE (PARA OLLAMA / LM STUDIO)")
print(f"📥 Origem: {CKPT_FILE} ({CKPT_FILE.stat().st_size / (1024**3):.2f} GB)")
print(f"💾 Destino: {OUTPUT_GGUF}")
print("=" * 72)

# 1. Carrega Tokenizador
print("📖 Carregando vocabulário do tokenizador BPE...")
with open(TOKENIZER_FILE, "r", encoding="utf-8") as f:
    tok_data = json.load(f)

vocab_dict = tok_data.get("model", {}).get("vocab", {})
merges = tok_data.get("model", {}).get("merges", [])

# Ordena tokens pelo ID
tokens_sorted = sorted(vocab_dict.items(), key=lambda x: x[1])
tokens_bytes = []
for tok_str, idx in tokens_sorted:
    tokens_bytes.append(tok_str.encode("utf-8", errors="replace"))

# 2. Inicializa o GGUF Writer para arquitetura LLaMA (compatível com MoE estilo Mixtral)
writer = gguf.GGUFWriter(str(OUTPUT_GGUF), "llama")

# Metadados da Arquitetura
writer.add_name("Synapse-1B-R1-MoE")
writer.add_description("Synapse 1.2B Sparse MoE (12 experts, Top-2) criada do zero por Nandinho.")
writer.add_context_length(1024)
writer.add_embedding_length(768)
writer.add_block_count(20)
writer.add_feed_forward_length(2048)
writer.add_expert_feed_forward_length(2048)
writer.add_head_count(12)
writer.add_head_count_kv(12)
writer.add_expert_count(12)
writer.add_expert_used_count(2)
writer.add_rope_freq_base(10000.0)
writer.add_rope_dimension_count(64)
writer.add_layer_norm_rms_eps(1e-5)

# Metadados do Tokenizador
writer.add_tokenizer_model("gpt2")
writer.add_token_list(tokens_bytes)
writer.add_token_scores([0.0] * len(tokens_bytes))
writer.add_token_types([1] * len(tokens_bytes))

if merges:
    merges_bytes = []
    for m in merges:
        if isinstance(m, list):
            s = f"{m[0]} {m[1]}"
        else:
            s = str(m)
        merges_bytes.append(s.encode("utf-8", errors="replace"))
    writer.add_token_merges(merges_bytes)

# IDs especiais
writer.add_bos_token_id(2)
writer.add_eos_token_id(2)
writer.add_pad_token_id(0)

print(f"✅ Vocabulário ({len(tokens_bytes):,} tokens) e metadados injetados com sucesso!")

# 3. Carrega os Pesos PyTorch
print(f"📥 Lendo pesos do PyTorch ({CKPT_FILE.name})...")
t0 = time.time()
raw = torch.load(str(CKPT_FILE), map_location="cpu", weights_only=False)
state = raw.get("model_state_dict", raw.get("model_state", raw))

# Limpa prefixos
clean = {}
for k, v in state.items():
    if k.startswith("_orig_mod."):
        clean[k[len("_orig_mod."):]] = v
    else:
        clean[k] = v

print(f"✅ Pesos lidos em {time.time()-t0:.2f}s! Mapeando tensores para formato GGUF FP16...")

# 4. Mapeamento dos Tensores
# Embedding
emb = clean["embedding.weight"].float().numpy().astype(np.float16)
writer.add_tensor("token_embd.weight", emb)

# RMSNorm final
norm = clean["norm.weight"].float().numpy()
writer.add_tensor("output_norm.weight", norm)

# Output Head (Tied)
head = clean["head.weight"].float().numpy().astype(np.float16)
writer.add_tensor("output.weight", head)

def permute_for_llama(w, n_head=12):
    # Converte o RoPE estilo HuggingFace/NeoX (intercalado 0::2, 1::2)
    # para a convenção nativa LLaMA no llama.cpp (metades contíguas)
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, -1).swapaxes(1, 2).reshape(w.shape)

# Camadas (20 Blocos MoE)
for b in range(20):
    pfx = f"blocks.{b}."
    
    # Norm 1
    w_norm1 = clean[f"{pfx}norm1.weight"].float().numpy()
    writer.add_tensor(f"blk.{b}.attn_norm.weight", w_norm1)
    
    # Attention QKV Chunked
    qkv = clean[f"{pfx}attn.qkv.weight"].float().numpy().astype(np.float16)
    q, k, v = np.split(qkv, 3, axis=0)
    writer.add_tensor(f"blk.{b}.attn_q.weight", permute_for_llama(q, 12))
    writer.add_tensor(f"blk.{b}.attn_k.weight", permute_for_llama(k, 12))
    writer.add_tensor(f"blk.{b}.attn_v.weight", v)
    
    # Attention Output
    w_out = clean[f"{pfx}attn.out.weight"].float().numpy().astype(np.float16)
    writer.add_tensor(f"blk.{b}.attn_output.weight", w_out)
    
    # Norm 2
    w_norm2 = clean[f"{pfx}norm2.weight"].float().numpy()
    writer.add_tensor(f"blk.{b}.ffn_norm.weight", w_norm2)
    
    # Router (Gate Input)
    router = clean[f"{pfx}moe.router.weight"].float().numpy()
    writer.add_tensor(f"blk.{b}.ffn_gate_inp.weight", router)
    
    # Especialistas Stacked: Gate, Up, Down (12 Especialistas por camada)
    # Forma esperada: (12, hidden_dim, dim)
    gates = [clean[f"{pfx}moe.experts.{e}.gate.weight"].float().numpy().astype(np.float16) for e in range(12)]
    ups = [clean[f"{pfx}moe.experts.{e}.up.weight"].float().numpy().astype(np.float16) for e in range(12)]
    downs = [clean[f"{pfx}moe.experts.{e}.down.weight"].float().numpy().astype(np.float16) for e in range(12)]
    
    writer.add_tensor(f"blk.{b}.ffn_gate_exps.weight", np.stack(gates, axis=0))
    writer.add_tensor(f"blk.{b}.ffn_up_exps.weight", np.stack(ups, axis=0))
    writer.add_tensor(f"blk.{b}.ffn_down_exps.weight", np.stack(downs, axis=0))
    
    print(f" -> Bloco [{b+1}/20] mapeado...")

# 5. Escreve o Arquivo GGUF
print("\n💾 Gravando arquivo GGUF final no disco...")
t_write = time.time()
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f"✨ CONVERSÃO CONCLUÍDA EM {time.time()-t_write:.2f}s!")
print(f"📁 Arquivo gerado: {OUTPUT_GGUF} ({OUTPUT_GGUF.stat().st_size / (1024**3):.2f} GB)")

# 6. Cria o Modelfile para o Ollama
MODELFILE_PATH = WORKSPACE / "Modelfile"
modelfile_content = f"""FROM ./{OUTPUT_GGUF.name}

PARAMETER temperature 0.6
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER stop "<|eos|>"
PARAMETER stop "<eos>"
PARAMETER stop "### Usuário:"
PARAMETER stop "### Assistente:"

TEMPLATE \"\"\"{{{{ if .Prompt }}}}### Usuário:
{{{{ .Prompt }}}}

### Assistente:
{{{{ end }}}}\"\"\"
"""

with open(MODELFILE_PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(modelfile_content)
print(f"✅ Modelfile gerado em: {MODELFILE_PATH}")
print("\n" + "=" * 72)
print("🚀 PARA CARREGAR NO OLLAMA, EXECUTE:")
print(f"   cd /d {WORKSPACE}")
print("   ollama create synapse -f Modelfile")
print("   ollama run synapse")
print("=" * 72)
