#!/usr/bin/env python3
"""
Auditoria técnica de qualidade e consistência de dados do Synapse.
Analisa integridade de JSONL, comprimentos, distribuição temática e vazamento de identidade.
"""

import json
import os
import sys
from collections import Counter

def auditar_arquivo(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado.")
        return

    print("=" * 65)
    print(f"RELATÓRIO DE AUDITORIA DE DATASET: {os.path.basename(caminho_arquivo)}")
    print("=" * 65)

    total_linhas = 0
    linhas_validas = 0
    linhas_corrompidas = 0
    
    subreddits = Counter()
    tamanhos_pergunta = []
    tamanhos_pensamento = []
    tamanhos_resposta = []
    
    palavras_bloqueadas = ["openai", "chatgpt", "gemini", "google", "deepseek", "anthropic", "claude"]
    vazamentos_identidade = []
    
    exemplos_amostra = []

    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        for idx, linha in enumerate(f, 1):
            total_linhas += 1
            linha_str = linha.strip()
            if not linha_str:
                continue
            try:
                d = json.loads(linha_str)
                p = d.get("pergunta", "").strip()
                t = d.get("pensamento", "").strip()
                r = d.get("resposta", "").strip()
                sub = d.get("subreddit", "desconhecido")
                
                if not p or not r:
                    linhas_corrompidas += 1
                    continue
                
                linhas_validas += 1
                subreddits[sub] += 1
                
                len_p = len(p.split())
                len_t = len(t.split())
                len_r = len(r.split())
                
                tamanhos_pergunta.append(len_p)
                tamanhos_pensamento.append(len_t)
                tamanhos_resposta.append(len_r)
                
                # Checagem de vazamento de identidade concorrente
                texto_completo = f"{p} {t} {r}".lower()
                for pb in palavras_bloqueadas:
                    if pb in texto_completo:
                        vazamentos_identidade.append((idx, pb, sub))
                
                if len(exemplos_amostra) < 3:
                    exemplos_amostra.append((sub, p, t, r))
                    
            except Exception as e:
                linhas_corrompidas += 1

    print(f"Total de linhas brutas:      {total_linhas}")
    print(f"Diálogos válidos:           {linhas_validas}")
    print(f"Linhas corrompidas/nulas:   {linhas_corrompidas}")
    print("-" * 65)
    
    print("DISTRIBUIÇÃO POR TEMA / SUBREDDIT:")
    for sub, count in subreddits.most_common():
        pct = (count / linhas_validas) * 100 if linhas_validas else 0
        print(f"  - {sub:<22}: {count:>5} amostras ({pct:.1f}%)")
    print("-" * 65)
    
    if tamanhos_pergunta:
        avg_p = sum(tamanhos_pergunta) / len(tamanhos_pergunta)
        avg_t = sum(tamanhos_pensamento) / len(tamanhos_pensamento)
        avg_r = sum(tamanhos_resposta) / len(tamanhos_resposta)
        print("MÉTRICAS DE COMPRIMENTO (MÉDIA DE PALAVRAS):")
        print(f"  - Pergunta do usuário:   {avg_p:.1f} palavras")
        print(f"  - Modo pensar (<think>): {avg_t:.1f} palavras")
        print(f"  - Resposta final:        {avg_r:.1f} palavras")
    print("-" * 65)

    print(f"VAZAMENTOS DE IDENTIDADE DETECTADOS: {len(vazamentos_identidade)}")
    if vazamentos_identidade:
        for idx, pb, sub in vazamentos_identidade[:5]:
            print(f"  - Linha {idx} [{sub}]: citação indevida a '{pb}'")
    else:
        print("  Zero vazamento. Nenhuma citação a Gemini, ChatGPT ou OpenAI.")
    print("-" * 65)

    print("AMOSTRA DE DADOS MINERADOS:\n")
    for i, (sub, p, t, r) in enumerate(exemplos_amostra, 1):
        print(f"--- [Amostra {i} | {sub}] ---")
        print(f"Usuário: {p}")
        print(f"<think>\n{t}\n</think>")
        print(f"Resposta: {r}\n")
    print("=" * 65)

if __name__ == "__main__":
    arquivo = sys.argv[1] if len(sys.argv) > 1 else "dataset_minerado_celular.jsonl"
    auditar_arquivo(arquivo)
