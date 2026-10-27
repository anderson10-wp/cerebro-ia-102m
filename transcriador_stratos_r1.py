#!/usr/bin/env python3
"""
TRANSCRIADOR COGNITIVO DE ELITE (HUGGING FACE STRATOS-17K -> SYNAPSE R1 PT-BR)
Desenvolvido por Nandinho.

Baixa problemas autênticos de olimpíadas científicas e matemática do dataset global
'HuggingFaceH4/Bespoke-Stratos-17k' e realiza Transcriação Cognitiva:
1. Problema adaptado com fluência nativa para o Português do Brasil.
2. Raciocínio re-executado do zero em 4 fases obrigatórias ([Entendimento], [Desenvolvimento], [Auto-Verificação], [Síntese]).
3. Auto-verificação com contra-prova matemática e lógica.
4. Gravação contínua no dataset_synapse_r1_reasoning.jsonl.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
from pathlib import Path

# Suporte UTF-8 no Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
KEY_GROQ = WORKSPACE / ".groq_key"
OUTPUT_FILE = WORKSPACE / "dataset_synapse_r1_reasoning.jsonl"
DESTILADO_FILE = WORKSPACE / "dataset_synapse_r1_destilado.jsonl"
PROGRESS_FILE = WORKSPACE / ".stratos_offset"

if not KEY_GROQ.exists():
    print("❌ Chave Groq (.groq_key) não encontrada!")
    sys.exit(1)

GROQ_KEY = KEY_GROQ.read_text(encoding="utf-8").strip()

# Offset para continuar de onde parou no Hugging Face
current_offset = 0
if PROGRESS_FILE.exists():
    try:
        current_offset = int(PROGRESS_FILE.read_text(encoding="utf-8").strip())
    except:
        current_offset = 0

perguntas_existentes = set()
if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                d = json.loads(line)
                p = d.get("pergunta", "").strip().lower()
                if p: perguntas_existentes.add(p)
            except:
                pass

print("=" * 72)
print("🧠 TRANSCRIADOR COGNITIVO SYNAPSE — BESPOKE-STRATOS (HUGGING FACE)")
print(f"📦 Amostras mestre existentes: {len(perguntas_existentes):,}")
print(f"📍 Posição inicial no Hugging Face: {current_offset}")
print(f"⚡ Motor de Transcriação: Groq LPU (Qwen 2.5 27B)")
print("=" * 72)

PROMPT_SISTEMA = """Você é um pesquisador de IA de elite especialista em Transcriação Cognitiva de Datasets (padrão DeepSeek-R1 e OpenAI o1).
Sua missão:
1. Pegar o problema em inglês do dataset Bespoke-Stratos e adaptá-lo para Português do Brasil com naturalidade e clareza impecáveis.
2. Re-raciocinar a solução do zero em português nas 4 etapas obrigatórias da Synapse dentro do pensamento (<think>):
   [Entendimento]: Decomposição do problema, identificação das grandezas, premissas e incógnitas.
   [Desenvolvimento]: Resolução matemática/lógica passo a passo demonstrando cada cálculo e dedução.
   [Auto-Verificação]: Simulação e teste de consistência ("Espere, deixe-me conferir se há furos ou erros nos cálculos..."). Faça a prova real.
   [Síntese]: Conclusão irrefutável antes de entregar a resposta final.
3. Resposta final didática, clara e direta.

Retorne APENAS um objeto JSON válido, sem texto fora dele, no formato:
{
  "categoria": "olimpiada_stratos",
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

def buscar_lote_huggingface(offset, limit=20):
    url = f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4%2FBespoke-Stratos-17k&config=default&split=train&offset={offset}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("rows", [])
    except Exception as e:
        print(f"[AVISO] Falha ao conectar no Hugging Face: {e}")
        return []

def transcriar_exemplo(raw_user, raw_assistant):
    payload = {
        "model": "qwen/qwen3.8-27b",
        "messages": [
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": f"PROBLEMA ORIGINAL EM INGLÊS:\n{raw_user}\n\nSOLUÇÃO DE REFERÊNCIA:\n{raw_assistant[:1400]}"}
        ],
        "temperature": 0.5,
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
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            dt = time.time() - t0
            raw = data["choices"][0]["message"]["content"]
            obj = extrair_json_limpo(raw)
            p = obj.get("pergunta", "").strip()
            t = obj.get("pensamento", "").strip()
            r = obj.get("resposta", "").strip()
            if p and t and r and len(p) >= 15 and len(t) >= 50:
                return {
                    "categoria": "olimpiada_stratos",
                    "pergunta": p,
                    "pensamento": t,
                    "resposta": r,
                    "treino_formatado": f"### Usuário:\n{p}\n\n### Assistente:\n<think>\n{t}\n</think>\n{r}",
                    "origem": "Bespoke-Stratos-17k",
                    "dt": dt
                }
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("[PAUSA] Limite de tokens da Groq atingido. Aguardando 15s...")
            time.sleep(15)
        else:
            print(f"[ERRO HTTP] {e.code}")
    except Exception as e:
        pass
    return None

def main():
    global current_offset
    total_gravados = len(perguntas_existentes)
    sucessos_sessao = 0
    t_inicio = time.time()

    print(f"🎯 Iniciando Transcriação de Alta Fidelidade...")
    print(f"⚡ Cadência: ~1 exemplo a cada 10s para respeitar o teto livre da Groq.")
    print("-" * 72)

    while True:
        rows = buscar_lote_huggingface(current_offset, limit=20)
        if not rows:
            print("[INFO] Fim do lote ou aguardando conexão...")
            time.sleep(10)
            continue

        for r in rows:
            current_offset += 1
            PROGRESS_FILE.write_text(str(current_offset), encoding="utf-8")
            
            row_data = r.get("row", {})
            conversations = row_data.get("conversations", [])
            if len(conversations) < 2:
                continue
                
            raw_user = conversations[0].get("value", "")
            raw_assistant = conversations[1].get("value", "")
            
            # Pula problemas triviais ou excessivamente longos
            if len(raw_user) < 20 or len(raw_user) > 2000:
                continue

            # Realiza a transcriação cognitiva
            amostra = transcriar_exemplo(raw_user, raw_assistant)
            
            if amostra:
                p_norm = amostra["pergunta"].strip().lower()
                if p_norm not in perguntas_existentes:
                    perguntas_existentes.add(p_norm)
                    
                    # Salva no arquivo destilado e no mestre
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_mestre:
                        f_mestre.write(json.dumps(amostra, ensure_ascii=False) + "\n")
                        f_mestre.flush()
                        
                    with open(DESTILADO_FILE, "a", encoding="utf-8") as f_dest:
                        f_dest.write(json.dumps(amostra, ensure_ascii=False) + "\n")
                        f_dest.flush()
                        
                    total_gravados += 1
                    sucessos_sessao += 1
                    t_decorrido = time.time() - t_inicio
                    taxa = (sucessos_sessao / max(1, t_decorrido)) * 60
                    
                    print(f"🌟 [STRATOS R1] Total: {total_gravados:>5} | +{sucessos_sessao} | {taxa:.1f}/min | dt: {amostra['dt']:.1f}s | Perg: {amostra['pergunta'][:50]}...")
            
            # Cadência anti-429 calibrada para Groq
            time.sleep(10.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[PAUSA] Transcriação pausada. Todas as amostras e o progresso foram salvos!")
