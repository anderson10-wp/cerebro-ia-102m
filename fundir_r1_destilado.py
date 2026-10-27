#!/usr/bin/env python3
"""
FUSÃO DE DATASETS R1 — PROJETO SYNAPSE
Mescla as amostras mineradas de alta fidelidade (OpenRouter + Groq)
diretamente na base mestra de alinhamento SFT: dataset_synapse_r1_reasoning.jsonl
"""

import sys
import json
from pathlib import Path

# Suporte UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
MINERADO = WORKSPACE / "dataset_synapse_r1_destilado.jsonl"
MESTRE = WORKSPACE / "dataset_synapse_r1_reasoning.jsonl"

def fundir():
    if not MINERADO.exists():
        print(f"Arquivo {MINERADO.name} não encontrado.")
        return

    perguntas_existentes = set()
    total_mestre_antes = 0

    if MESTRE.exists():
        with open(MESTRE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                total_mestre_antes += 1
                try:
                    d = json.loads(line)
                    perguntas_existentes.add(d.get("pergunta", "").strip().lower())
                except:
                    pass

    novas_adicionadas = 0
    with open(MINERADO, "r", encoding="utf-8") as f_in, open(MESTRE, "a", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                p = item.get("pergunta", "").strip().lower()
                if p and p not in perguntas_existentes:
                    perguntas_existentes.add(p)
                    # Garante formato padronizado
                    if "treino_formatado" not in item:
                        p_txt = item["pergunta"]
                        t_txt = item["pensamento"]
                        r_txt = item["resposta"]
                        item["treino_formatado"] = f"### Usuário:\n{p_txt}\n\n### Assistente:\n<think>\n{t_txt}\n</think>\n{r_txt}"
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    novas_adicionadas += 1
            except Exception:
                pass

    print("=" * 60)
    print("🔄 FUSÃO DE DADOS R1 CONCLUÍDA")
    print(f"📦 Amostras mestre antes: {total_mestre_antes:,}")
    print(f"✨ Novas amostras fundidas: +{novas_adicionadas:,}")
    print(f"🚀 Total consolidado agora: {total_mestre_antes + novas_adicionadas:,}")
    print("=" * 60)

if __name__ == "__main__":
    fundir()
