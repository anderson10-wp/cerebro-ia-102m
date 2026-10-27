#!/usr/bin/env python3
"""
PREPARADOR DE DADOS PARA SYNAPSE 1B (PIPELINE BINÁRIO MEMMAP)
Baixa, limpa e compila grandes massas de texto e código diretamente no Google Drive (5 TB)
em formato binário uint16 (2 bytes por token) para treinamento sem gargalos de I/O.

Fontes integradas:
1. Alpaca GPT-4 Português (50k conversas instrutivas de alta precisão)
2. Wikipédia em Português Completa (wikimedia/wikipedia - 20231101.pt)
3. Python Code Repositories (Código limpo do The Stack / BigCode)
Saída: /content/drive/MyDrive/Synapse-1B/dados/treino_1b.bin
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np

# Suporte UTF-8 no Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

from tokenizers import Tokenizer

def main():
    parser = argparse.ArgumentParser(description="Compilador Binário de Pré-Treino para Synapse 1B")
    parser.add_argument("--limite_docs_wiki", type=int, default=50000, help="Quantidade de artigos da Wikipédia a processar")
    parser.add_argument("--limite_docs_code", type=int, default=30000, help="Quantidade de arquivos de código a processar")
    parser.add_argument("--local", action="store_true", help="Salvar na pasta local em vez do Google Drive")
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 SYNAPSE 1B — COMPILADOR DE DADOS DE PRÉ-TREINO (MEMMAP uint16)")
    print("=" * 70)

    # 1. Definir caminhos de armazenamento
    if args.local:
        DRIVE_ROOT = Path("dados_synapse_1b")
    else:
        COLAB_DRIVE = Path("/content/drive/MyDrive/Synapse-1B")
        if not Path("/content/drive/MyDrive").exists():
            print("⚠️ Google Drive não detectado em /content/drive/MyDrive.")
            print("   Usando pasta local 'dados_synapse_1b'...")
            DRIVE_ROOT = Path("dados_synapse_1b")
        else:
            DRIVE_ROOT = COLAB_DRIVE

    PASTA_DADOS = DRIVE_ROOT / "dados"
    PASTA_CHECKPOINTS = DRIVE_ROOT / "checkpoints"
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    PASTA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    ARQUIVO_BIN = PASTA_DADOS / "treino_1b.bin"
    ARQUIVO_META = PASTA_DADOS / "meta_dados.json"

    print(f"📁 Diretório de Dados no Drive:       {PASTA_DADOS}")
    print(f"📁 Diretório de Checkpoints no Drive:  {PASTA_CHECKPOINTS}")
    print(f"💾 Arquivo Binário de Destino:        {ARQUIVO_BIN}")
    print("-" * 70)

    # 2. Carregar Tokenizador Oficial (32k)
    tok_candidates = [
        Path("tokenizer.json"),
        Path("/content/cerebro-ia-102m/tokenizer.json"),
        Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m\tokenizer.json"),
    ]
    tok_file = None
    for cand in tok_candidates:
        if cand.exists():
            tok_file = cand
            break
    if not tok_file:
        cand_list = list(Path(".").rglob("tokenizer.json"))
        if cand_list:
            tok_file = cand_list[0]

    if not tok_file:
        print("❌ Erro: 'tokenizer.json' não encontrado!")
        sys.exit(1)

    print(f"✅ Tokenizador Carregado: {tok_file}")
    tok = Tokenizer.from_file(str(tok_file))
    eos_id = tok.token_to_id("<eos>") or 2

    # 3. Buffer de Escrita Incremental em Disco
    BUFFER_SIZE = 250_000  # descarrega a cada 250k tokens (500 KB) para não estourar RAM
    token_buffer = []
    total_tokens_gravados = 0
    t0 = time.time()

    # Modo de abertura: append se já existir para permitir retomada
    if ARQUIVO_BIN.exists() and ARQUIVO_META.exists():
        try:
            with open(ARQUIVO_META, "r", encoding="utf-8") as f:
                meta = json.load(f)
                total_tokens_gravados = meta.get("total_tokens", 0)
            print(f"🔄 Arquivo existente detectado com {total_tokens_gravados:,} tokens já gravados.")
            print("   Continuando processo em modo APPEND...")
        except:
            total_tokens_gravados = ARQUIVO_BIN.stat().st_size // 2

    def flush_buffer():
        nonlocal total_tokens_gravados
        if not token_buffer:
            return
        arr = np.array(token_buffer, dtype=np.uint16)
        with open(ARQUIVO_BIN, "ab") as f_out:
            f_out.write(arr.tobytes())
        total_tokens_gravados += len(token_buffer)
        token_buffer.clear()
        
        # Salva metadados atualizados
        t_decorrido = time.time() - t0
        tok_por_seg = total_tokens_gravados / max(1, t_decorrido)
        tamanho_mb = (total_tokens_gravados * 2) / (1024 * 1024)
        with open(ARQUIVO_META, "w", encoding="utf-8") as f_meta:
            json.dump({
                "total_tokens": total_tokens_gravados,
                "tamanho_mb": round(tamanho_mb, 2),
                "dtype": "uint16",
                "vocab_size": tok.get_vocab_size()
            }, f_meta, indent=2)
            
        print(f"💾 Progresso: {total_tokens_gravados:,} tokens gravados | {tamanho_mb:.2f} MB | {tok_por_seg:.0f} tok/s", end="\r")

    # -------------------------------------------------------------------------
    # PARTE 1: ALPACA GPT-4 EM PORTUGUÊS (50k amostras instrutivas)
    # -------------------------------------------------------------------------
    alpaca_candidates = [
        Path("alpaca-gpt4-portuguese.json"),
        Path("/content/cerebro-ia-102m/alpaca-gpt4-portuguese.json"),
        Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m\alpaca-gpt4-portuguese.json")
    ]
    alpaca_file = None
    for cand in alpaca_candidates:
        if cand.exists():
            alpaca_file = cand
            break

    if alpaca_file:
        print("\n[1/3] Processando Alpaca GPT-4 Português...")
        with open(alpaca_file, "r", encoding="utf-8") as f:
            alpaca_data = json.load(f)
            
        for item in alpaca_data:
            convs = item.get("conversations", [])
            if len(convs) < 2: continue
            texto = f"### Usuário:\n{convs[0]['value'].strip()}\n\n### Assistente:\n{convs[1]['value'].strip()}"
            ids = tok.encode(texto).ids
            token_buffer.extend(ids)
            token_buffer.append(eos_id)
            if len(token_buffer) >= BUFFER_SIZE:
                flush_buffer()
        flush_buffer()
        print(f"\n   -> Alpaca GPT-4 PT processado! Total: {total_tokens_gravados:,} tokens.")

    # -------------------------------------------------------------------------
    # PARTE 2: WIKIPÉDIA EM PORTUGUÊS (wikimedia/wikipedia)
    # -------------------------------------------------------------------------
    print(f"\n[2/3] Baixando e tokenizando Wikipédia em Português ({args.limite_docs_wiki:,} artigos)...")
    try:
        from datasets import load_dataset
        ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
        count_wiki = 0
        for doc in ds_wiki:
            texto = doc.get("text", "").strip()
            if len(texto) < 100: continue
            ids = tok.encode(texto).ids
            token_buffer.extend(ids)
            token_buffer.append(eos_id)
            count_wiki += 1
            if len(token_buffer) >= BUFFER_SIZE:
                flush_buffer()
            if count_wiki >= args.limite_docs_wiki:
                break
        flush_buffer()
        print(f"\n   -> {count_wiki:,} artigos da Wikipédia gravados! Total acumulado: {total_tokens_gravados:,} tokens.")
    except Exception as e:
        print(f"\n⚠️ Aviso ao carregar Wikipédia: {e}")
        print("   (Verifique conexão ou execute `pip install datasets` no Colab)")

    # -------------------------------------------------------------------------
    # PARTE 3: PYTHON CODE REPOSITORIES (The Stack / BigCode)
    # -------------------------------------------------------------------------
    print(f"\n[3/3] Baixando e tokenizando Repositórios de Código Python ({args.limite_docs_code:,} arquivos)...")
    try:
        from datasets import load_dataset
        ds_code = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
        count_code = 0
        for doc in ds_code:
            code = doc.get("content", "").strip()
            if len(code) < 50 or len(code) > 50000: continue
            # Evita arquivos minificados ou binários
            if "\x00" in code: continue
            ids = tok.encode(code).ids
            token_buffer.extend(ids)
            token_buffer.append(eos_id)
            count_code += 1
            if len(token_buffer) >= BUFFER_SIZE:
                flush_buffer()
            if count_code >= args.limite_docs_code:
                break
        flush_buffer()
        print(f"\n   -> {count_code:,} arquivos Python gravados! Total acumulado: {total_tokens_gravados:,} tokens.")
    except Exception as e:
        print(f"\n⚠️ Aviso ao carregar The Stack Python: {e}")

    flush_buffer()
    tamanho_final_gb = (total_tokens_gravados * 2) / (1024**3)
    tempo_total = time.time() - t0

    print("\n" + "=" * 70)
    print("✅ COMPILAÇÃO DO DATASET DE PRÉ-TREINO 1B CONCLUÍDA COM SUCESSO!")
    print(f"📁 Localização:   {ARQUIVO_BIN}")
    print(f"📊 Total Tokens:  {total_tokens_gravados:,} tokens gravados")
    print(f"💾 Tamanho Real:  {tamanho_final_gb:.2f} GB (uint16 Memmap)")
    print(f"⏱️ Tempo Total:   {tempo_total:.1f} segundos")
    print("=" * 70)

if __name__ == "__main__":
    main()
