#!/usr/bin/env python3
"""
MINERADOR INDUSTRIAL DE RACIOCÍNIO R1 VIA OPENROUTER (PROFESSOR GPT-4o-MINI)
Desenvolvido por Nandinho.
Utiliza a API do OpenRouter para gerar milhares de exemplos de altíssima fidelidade
com rastros de raciocínio profundo (<think>) estruturados:
- [Entendimento do Problema]
- [Decomposição & Hipóteses]
- [Auto-Verificação & Aha Moment]
- [Conclusão & Síntese]

Salva em: dataset_synapse_r1_destilado.jsonl
"""

import os
import sys
import time
import json
import random
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suporte UTF-8 no Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
KEY_FILE = WORKSPACE / ".openrouter_key"
OUTPUT_FILE = WORKSPACE / "dataset_synapse_r1_destilado.jsonl"
PROGRESS_FILE = WORKSPACE / "meta_mineracao_r1.json"

if not KEY_FILE.exists():
    print(f"❌ Erro: Arquivo de chave '.openrouter_key' não encontrado em: {KEY_FILE}")
    sys.exit(1)

API_KEY = KEY_FILE.read_text(encoding="utf-8").strip()

# Categorias e temas para variedade máxima
TEMAS = [
    {
        "cat": "matematica_complexa",
        "desc": "Problemas de álgebra, porcentagem, equações, frações, probabilidade ou problemas verbais com pegadinhas numéricas."
    },
    {
        "cat": "algoritmos_python",
        "desc": "Implementação de funções, algoritmos de busca/ordenação, manipulação de strings/listas/dicionários, decorators, generators ou tratamento de exceções em Python."
    },
    {
        "cat": "ciencia_e_fisica",
        "desc": "Conceitos de física (gravidade, termodinâmica, velocidade), química, biologia humana, astronomia ou geografia de precisão."
    },
    {
        "cat": "logica_e_deducao",
        "desc": "Enigmas de lógica, dedução de culpados, sequência de padrões, paradoxos lógicos ou análise de premissas verdadeiras/falsas."
    },
    {
        "cat": "identidade_synapse",
        "desc": "Perguntas sobre a inteligência artificial Synapse, desenvolvida do zero pelo Nandinho com arquitetura Sparse MoE de 1.2B e foco em soberania local."
    },
    {
        "cat": "dialogo_e_conselhos",
        "desc": "Conselhos sobre desenvolvimento pessoal, rotina de estudos, como aprender a programar, como fazer amizades e inteligência emocional com maturidade."
    }
]

perguntas_existentes = set()
if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                d = json.loads(line)
                perguntas_existentes.add(d["pergunta"].strip().lower())
            except:
                pass

print("=" * 70)
print(f"🚀 MINERADOR R1 OPENROUTER ATIVADO | Base existente: {len(perguntas_existentes):,} amostras")
print(f"💾 Destino: {OUTPUT_FILE}")
print("=" * 70)

def gerar_amostra_api(tema_info):
    prompt_sistema = """Você é um especialista em destilação de raciocínio para modelos de inteligência artificial de ponta (padrão DeepSeek-R1 e OpenAI o1).
Sua missão é gerar 1 exemplo INÉDITO, desafiador e educativo no idioma Português do Brasil com um rastro de pensamento interno altamente detalhado.

Regras estritas para o pensamento (<think>):
- O pensamento DEVE ter pelo menos 3 a 5 parágrafos organizados nas 4 etapas:
  [Entendimento]: Decompor o pedido, identificar armadilhas ou ambiguidades.
  [Desenvolvimento]: Resolver passo a passo, demonstrando os cálculos e hipóteses.
  [Auto-Verificação]: Parar e conferir o resultado ("Espere, deixe-me verificar se há erros..."). Simular casos de teste.
  [Síntese]: Concluir de forma irrefutável antes de fechar o pensamento.
- A resposta final deve ser limpa, didática e direta.
- Retorne APENAS um objeto JSON válido, sem texto antes ou depois, no formato:
{
  "categoria": "...",
  "pergunta": "...",
  "pensamento": "...",
  "resposta": "..."
}"""

    prompt_usuario = f"""Gere um exemplo para a categoria '{tema_info['cat']}':
Foco: {tema_info['desc']}
Gere uma pergunta criativa e natural que um usuário real faria. Garanta que o pensamento seja profundo e mostre auto-correção mental."""

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario}
        ],
        "temperature": 0.85,
        "max_tokens": 1200
    }

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/anderson10-wp/cerebro-ia-102m",
            "X-Title": "Synapse-R1-Data-Engine"
        }
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            dt = time.time() - t0
            raw_text = data["choices"][0]["message"]["content"].strip()
            
            # Limpa possíveis blocos markdown ```json ... ```
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()
            
            obj = json.loads(raw_text)
            p = obj.get("pergunta", "").strip()
            t = obj.get("pensamento", "").strip()
            r = obj.get("resposta", "").strip()
            cat = obj.get("categoria", tema_info["cat"]).strip()
            
            if p and t and r and len(p) >= 10 and len(t) >= 40:
                return {
                    "categoria": cat,
                    "pergunta": p,
                    "pensamento": t,
                    "resposta": r,
                    "treino_formatado": f"### Usuário:\n{p}\n\n### Assistente:\n<think>\n{t}\n</think>\n{r}",
                    "dt": dt
                }
    except urllib.error.HTTPError as e:
        if e.code == 429:
            time.sleep(3.0)
        return None
    except Exception:
        return None
    return None

def main():
    total_gravados = len(perguntas_existentes)
    sucessos_sessao = 0
    t_inicio = time.time()
    MAX_ALVO = 5000  # Meta alta: até 5.000 amostras ou até esgotar cota

    print(f"⚡ Disparando mineração paralela com GPT-4o-mini...")
    print(f"🎯 Meta Alvo: até {MAX_ALVO} amostras de ouro R1.")
    print("-" * 70)

    # 4 Workers simultâneos para velocidade sem estourar limites de concorrência
    with ThreadPoolExecutor(max_workers=4) as executor:
        while total_gravados < MAX_ALVO:
            futures = [
                executor.submit(gerar_amostra_api, random.choice(TEMAS))
                for _ in range(8)
            ]
            
            for future in as_completed(futures):
                res = future.result()
                if res:
                    p_norm = res["pergunta"].strip().lower()
                    if p_norm not in perguntas_existentes:
                        perguntas_existentes.add(p_norm)
                        with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
                            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                            f_out.flush()
                        
                        total_gravados += 1
                        sucessos_sessao += 1
                        t_decorrido = time.time() - t_inicio
                        amostras_por_min = (sucessos_sessao / max(1, t_decorrido)) * 60
                        
                        print(f"✅ [+1 R1] Total: {total_gravados:>5} | Sessão: +{sucessos_sessao} | {amostras_por_min:.1f} amostras/min | Cat: {res['categoria']:<22} | dt: {res['dt']:.2f}s")
                    else:
                        pass
                else:
                    time.sleep(0.5)

            # Pequena pausa para manter requisições limpas
            time.sleep(0.2)

if __name__ == "__main__":
    main()
