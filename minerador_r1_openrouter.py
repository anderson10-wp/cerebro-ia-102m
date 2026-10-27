#!/usr/bin/env python3
"""
MINERADOR INDUSTRIAL MULTI-TURBO (OPENROUTER GPT-4o + GROQ LPU)
Projeto Synapse — Raciocínio Profundo Padrão DeepSeek-R1 / OpenAI o1.
Desenvolvido por Nandinho.

Capacidades:
- Execução contínua em larga escala (meta configurada para 10.000+ amostras)
- OpenRouter turbo multithreading (GPT-4o-mini)
- Groq LPU sincronizado com cadência anti-429 (Qwen 27B)
- Deduplicação em tempo real
- Gravação instantânea em dataset_synapse_r1_destilado.jsonl
"""

import os
import sys
import time
import json
import random
import threading
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Força UTF-8 no terminal Windows para evitar falhas de codificação
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
KEY_OPENROUTER = WORKSPACE / ".openrouter_key"
KEY_GROQ = WORKSPACE / ".groq_key"
OUTPUT_FILE = WORKSPACE / "dataset_synapse_r1_destilado.jsonl"
BASE_REASONING = WORKSPACE / "dataset_synapse_r1_reasoning.jsonl"

OPENROUTER_KEY = KEY_OPENROUTER.read_text(encoding="utf-8").strip() if KEY_OPENROUTER.exists() else None
GROQ_KEY = KEY_GROQ.read_text(encoding="utf-8").strip() if KEY_GROQ.exists() else None

if not OPENROUTER_KEY and not GROQ_KEY:
    print("[ERRO] Nenhuma chave de API detectada em .openrouter_key ou .groq_key!")
    sys.exit(1)

TEMAS = [
    {
        "cat": "matematica_complexa",
        "desc": "Problemas desafiadores de álgebra, porcentagem, equações, frações, probabilidade, trigonometria ou problemas verbais com pegadinhas numéricas."
    },
    {
        "cat": "algoritmos_python",
        "desc": "Implementação de funções, algoritmos de busca/ordenação, manipulação de dicionários/listas, decorators, generators, recursão ou tratamento de exceções em Python."
    },
    {
        "cat": "ciencia_e_fisica",
        "desc": "Conceitos de física clássica e moderna (gravidade, termodinâmica, velocidade relativa, eletromagnetismo), química e astronomia com rigor científico."
    },
    {
        "cat": "logica_e_deducao",
        "desc": "Enigmas de lógica dedutiva, identificação de falácias argumentativas, sequência de padrões complexos, paradoxos lógicos ou análise de premissas."
    },
    {
        "cat": "identidade_synapse",
        "desc": "Perguntas sobre a inteligência artificial Synapse, desenvolvida do zero pelo Nandinho com arquitetura Sparse MoE de 1.2B e foco em soberania local."
    },
    {
        "cat": "dialogo_e_conselhos",
        "desc": "Conselhos sobre desenvolvimento pessoal, rotina de estudos, como aprender a programar, produtividade, inteligência emocional e resolução de conflitos."
    },
    {
        "cat": "engenharia_e_hardware",
        "desc": "Arquitetura de computadores, funcionamento de VRAM, Barramento PCIe, DirectML, FP16 vs BF16, Quantização 4-bit e GPUs AMD Radeon."
    }
]

# Carrega perguntas existentes para não gerar duplicadas
perguntas_existentes = set()
lock_gravacao = threading.Lock()
lock_groq_tempo = threading.Lock()
ultimo_acesso_groq = 0.0

def carregar_existentes(caminho):
    if not caminho.exists(): return
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    d = json.loads(line)
                    p = d.get("pergunta", "").strip().lower()
                    if p: perguntas_existentes.add(p)
                except:
                    pass
    except Exception:
        pass

carregar_existentes(OUTPUT_FILE)
carregar_existentes(BASE_REASONING)

print("=" * 72)
print("🚀 MOTOR DE MINERAÇÃO R1 — PROVEDORES DE PONTA EM PARALELO")
print(f"📁 Amostras existentes na memória: {len(perguntas_existentes):,}")
print(f"🔑 Provedores: {'[OpenRouter GPT-4o]' if OPENROUTER_KEY else ''} {'[Groq Qwen-27B]' if GROQ_KEY else ''}")
print(f"💾 Arquivo de saída: {OUTPUT_FILE.name}")
print("=" * 72)

PROMPT_SISTEMA = """Você é um professor e pesquisador de Inteligência Artificial de elite, especialista em destilação de raciocínio profundo padrão DeepSeek-R1 e OpenAI o1.
Sua missão é gerar 1 exemplo INÉDITO, desafiador e educativo no idioma Português do Brasil com um rastro de pensamento interno (<think>) altamente analítico.

Regras estritas para o pensamento (<think>):
- O pensamento DEVE conter raciocínio detalhado dividido nas 4 etapas lógicas:
  [Entendimento]: Decompor a questão, identificar sutilezas ou armadilhas.
  [Desenvolvimento]: Resolver passo a passo com clareza conceitual ou matemática.
  [Auto-Verificação]: Questionar o resultado intermediário ("Espere, deixe-me conferir se há erros ou casos de borda...").
  [Síntese]: Concluir de forma irrefutável antes de redigir a resposta final.
- A resposta final deve ser clara, didática, polida e completa.
- Retorne APENAS um objeto JSON válido, sem texto antes ou depois, exatamente na estrutura:
{
  "categoria": "...",
  "pergunta": "...",
  "pensamento": "...",
  "resposta": "..."
}"""

def extrair_json_limpo(raw_text):
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    ini = text.find("{")
    fim = text.rfind("}")
    if ini != -1 and fim != -1:
        text = text[ini : fim + 1]
    return json.loads(text)

def minerar_openrouter(tema_info):
    if not OPENROUTER_KEY: return None
    prompt_usuario = f"Gere um exemplo desafiador para a categoria '{tema_info['cat']}': {tema_info['desc']}. Mostre pensamento com auto-verificação rigorosa."
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": prompt_usuario}
        ],
        "temperature": 0.85,
        "max_tokens": 1200
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "Synapse-R1-Engine/1.0"
        }
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            dt = time.time() - t0
            raw = data["choices"][0]["message"]["content"]
            obj = extrair_json_limpo(raw)
            p, t, r = obj.get("pergunta", "").strip(), obj.get("pensamento", "").strip(), obj.get("resposta", "").strip()
            cat = obj.get("categoria", tema_info["cat"]).strip()
            if p and t and r and len(p) >= 10 and len(t) >= 40:
                return {
                    "categoria": cat, "pergunta": p, "pensamento": t, "resposta": r,
                    "treino_formatado": f"### Usuário:\n{p}\n\n### Assistente:\n<think>\n{t}\n</think>\n{r}",
                    "motor": "GPT-4o-mini", "dt": dt
                }
    except Exception:
        return None
    return None

def minerar_groq(tema_info):
    global ultimo_acesso_groq
    if not GROQ_KEY: return None
    
    # Cadência de 10s para respeitar o limite de 8.000 TPM gratuito da Groq
    with lock_groq_tempo:
        agora = time.time()
        espera = 10.0 - (agora - ultimo_acesso_groq)
        if espera > 0:
            time.sleep(espera)
        ultimo_acesso_groq = time.time()

    prompt_usuario = f"Gere um exemplo desafiador para a categoria '{tema_info['cat']}': {tema_info['desc']}. Responda apenas com o JSON estrito."
    payload = {
        "model": "qwen/qwen3.8-27b",
        "messages": [
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": prompt_usuario}
        ],
        "temperature": 0.75,
        "max_tokens": 1200
    }
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {GROQ_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            dt = time.time() - t0
            raw = data["choices"][0]["message"]["content"]
            obj = extrair_json_limpo(raw)
            p, t, r = obj.get("pergunta", "").strip(), obj.get("pensamento", "").strip(), obj.get("resposta", "").strip()
            cat = obj.get("categoria", tema_info["cat"]).strip()
            if p and t and r and len(p) >= 10 and len(t) >= 40:
                return {
                    "categoria": cat, "pergunta": p, "pensamento": t, "resposta": r,
                    "treino_formatado": f"### Usuário:\n{p}\n\n### Assistente:\n<think>\n{t}\n</think>\n{r}",
                    "motor": "Groq-Qwen27B", "dt": dt
                }
    except Exception:
        return None
    return None

def minerar_tarefa(tema_info):
    # Distribuição inteligente: 80% OpenRouter (sem limite de TPM) + 20% Groq (cadenciado)
    if OPENROUTER_KEY and GROQ_KEY:
        if random.random() < 0.25:
            res = minerar_groq(tema_info)
            if not res: res = minerar_openrouter(tema_info)
            return res
        else:
            return minerar_openrouter(tema_info)
    elif OPENROUTER_KEY:
        return minerar_openrouter(tema_info)
    else:
        return minerar_groq(tema_info)

def main():
    total_gravados = len(perguntas_existentes)
    sucessos_sessao = 0
    t_inicio = time.time()
    MAX_ALVO = 10000  # Meta expandida para capacidade máxima contínua

    print(f"🎯 Meta Máxima: {MAX_ALVO:,} amostras de raciocínio profundo R1.")
    print(f"⚡ Operação Multithread iniciada. Pressione CTRL+C para pausar a qualquer momento.")
    print("-" * 72)

    with ThreadPoolExecutor(max_workers=5) as executor:
        while total_gravados < MAX_ALVO:
            futures = [
                executor.submit(minerar_tarefa, random.choice(TEMAS))
                for _ in range(10)
            ]
            
            for future in as_completed(futures):
                try:
                    res = future.result()
                except Exception:
                    res = None

                if res:
                    p_norm = res["pergunta"].strip().lower()
                    with lock_gravacao:
                        if p_norm not in perguntas_existentes:
                            perguntas_existentes.add(p_norm)
                            with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
                                f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                                f_out.flush()
                            
                            total_gravados += 1
                            sucessos_sessao += 1
                            t_decorrido = time.time() - t_inicio
                            amostras_por_min = (sucessos_sessao / max(1, t_decorrido)) * 60
                            
                            print(f"[OK R1] Total: {total_gravados:>5} | +{sucessos_sessao} nesta sessão | {amostras_por_min:.1f}/min | [{res['motor']}] | Cat: {res['categoria']:<22} | dt: {res['dt']:.2f}s")
                else:
                    time.sleep(0.3)
            time.sleep(0.2)

    print("\n[CONCLUÍDO] Meta de mineração atingida com sucesso!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[PAUSA] Mineração pausada pelo usuário. Todas as amostras foram salvas com segurança no arquivo!")
