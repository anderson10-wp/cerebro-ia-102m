#!/usr/bin/env python3
"""
MINERADOR GROQ TURBO — CADÊNCIA CALIBRADA (ANTI-400 / ANTI-429)
Controla a taxa de Tokens Por Minuto (TPM) para operar 100% contínuo
abaixo do teto de 8.000 TPM da conta gratuita da Groq.
Compatível com Windows (UTF-8/cp1252), Linux e Termux.
"""

import os
import sys

# Configura UTF-8 imediato no console do Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import urllib.request
import urllib.error
import json
import time
import random

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-20b"
OUTPUT_FILE = "dataset_minerado_groq.jsonl"

TEMAS = [
    ("Python Sintaxe & Prática", "Diferença entre lista e tupla, dicionários, métodos de string, funções com def, try/except, loops for/while e list comprehensions com exemplos concisos de código."),
    ("Lógica & Resolução de Problemas", "Raciocínio lógico estruturado, tomada de decisão prática, resolução de problemas cotidianos e pensamento analítico."),
    ("Conversação Brasileira & Cotidiano", "Diálogos autênticos, histórias curiosas, conselhos de vida, relacionamentos, trabalho e rotina sem formalismo excessivo."),
    ("Ciência & Conhecimento Geral", "Explicações claras sobre biologia, física cotidiana, espaço, história e curiosidades do Brasil e do mundo."),
    ("Carreira & Tecnologia", "Como construir portfólio real, dicas para desenvolvedor júnior, boas práticas de código limpo e rotina de home office.")
]

def obter_chave():
    if os.environ.get("GROQ_API_KEY"):
        return os.environ.get("GROQ_API_KEY").strip()
    if os.path.exists(".groq_key"):
        with open(".groq_key", "r", encoding="utf-8") as f:
            return f.read().strip()
    
    print("\n[CHAVE] Nenhuma chave Groq encontrada.")
    k = input("Cole sua Chave da API Groq (gsk_...): ").strip()
    if k.startswith("gsk_"):
        try:
            with open(".groq_key", "w", encoding="utf-8") as f:
                f.write(k)
            print("[INFO] Chave salva em .groq_key para próximas execuções!\n")
        except Exception:
            pass
        return k
    print("[ERRO] Chave inválida! Deve começar com 'gsk_'.")
    sys.exit(1)

def gerar_lote_groq(api_key, tema_titulo, tema_desc):
    prompt_sistema = """Você é um sintetizador de dados para treinamento de inteligência artificial em português do Brasil.
Retorne SEMPRE e ESTRITAMENTE um objeto JSON com a chave "dialogos" contendo uma lista de 2 objetos.
Cada objeto DEVE ter os campos:
- "tema": título do tema
- "pergunta": pergunta realista e natural do usuário em português
- "pensamento": reflexão no modo pensar (<think>) planejando a resposta de forma clara, técnica e empática
- "resposta": resposta final completa e bem estruturada."""

    prompt_usuario = f"""Gere 2 diálogos concisos sobre: {tema_titulo} ({tema_desc}).
Retorne o JSON no formato:
{{
  "dialogos": [
    {{"tema": "{tema_titulo}", "pergunta": "...", "pensamento": "...", "resposta": "..."}},
    ...
  ]
}}"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario}
        ],
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
        "max_tokens": 800  # Limite rígido para evitar estouro de TPM e truncamento do JSON
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GROQ_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        # Telemetria de Rate Limit oficial da Groq
        remaining_tokens = resp.headers.get("x-ratelimit-remaining-tokens")
        reset_time_str = resp.headers.get("x-ratelimit-reset-tokens", "6s")
        
        # 1. Extrair resposta da API OpenAI/Groq
        api_res = json.loads(resp.read().decode("utf-8"))
        model_content = api_res["choices"][0]["message"]["content"]
        
        # 2. Fazer o parse do JSON gerado pelo modelo
        parsed = json.loads(model_content)
        
        dialogos = []
        if isinstance(parsed, dict):
            if "dialogos" in parsed and isinstance(parsed["dialogos"], list):
                dialogos = parsed["dialogos"]
            else:
                for v in parsed.values():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        dialogos = v
                        break
        elif isinstance(parsed, list):
            dialogos = parsed
            
        return dialogos, remaining_tokens, reset_time_str

def parse_reset_seconds(s):
    try:
        s = s.strip().lower()
        if "ms" in s:
            return float(s.replace("ms", "")) / 1000.0
        if "s" in s:
            return float(s.replace("s", ""))
        return 6.0
    except:
        return 6.0

def main():
    print("=" * 70)
    print("MINERADOR GROQ TURBO — FLUXO ESTAVEL (ANTI-400 / ANTI-429)")
    print(f"Arquivo de Saida: {OUTPUT_FILE}")
    print(f"Modelo Base: {MODEL_NAME} (Lote: 2 amostras | max_tokens: 800)")
    print("=" * 70)

    api_key = obter_chave()
    total = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            total = sum(1 for line in f if line.strip())

    print(f"Amostras ja existentes no arquivo: {total}\n")

    rodada = 1
    erros = 0

    while True:
        tema_titulo, tema_desc = random.choice(TEMAS)
        hora = time.strftime("%H:%M:%S")
        print(f"[{hora}] [Rodada {rodada}] Minerando 2 amostras de '{tema_titulo}'...")

        try:
            t0 = time.time()
            lote, remaining_tokens, reset_str = gerar_lote_groq(api_key, tema_titulo, tema_desc)
            dt = time.time() - t0
            novos = 0

            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for item in lote:
                    p = item.get("pergunta", "").strip()
                    t = item.get("pensamento", "").strip()
                    r = item.get("resposta", "").strip()
                    tema = item.get("tema", tema_titulo)

                    if p and r:
                        treino = f"### Usuário:\n{p}\n\n### Assistente:\n<think>\n{t}\n</think>\n{r}"
                        registro = {
                            "tema": tema,
                            "pergunta": p,
                            "pensamento": t,
                            "resposta": r,
                            "treino_formatado": treino
                        }
                        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
                        novos += 1
                        total += 1

            erros = 0
            rem_info = f" | Restam {remaining_tokens} tokens" if remaining_tokens else ""
            print(f"   [OK] +{novos} amostras gravadas em {dt:.2f}s! Total acumulado: {total}{rem_info}")
            rodada += 1

            # Gestão Adaptativa de Cota: se estiver com menos de 1.200 tokens livres, espera o reset
            if remaining_tokens:
                try:
                    if int(remaining_tokens) < 1200:
                        espera = parse_reset_seconds(reset_str) + 1.0
                        print(f"   [PAUSA] Cota quase cheia. Aguardando {espera:.1f}s para reset...")
                        time.sleep(espera)
                        continue
                except:
                    pass

            # Intervalo padrão seguro de 6.0 segundos
            time.sleep(6.0)

        except urllib.error.HTTPError as http_err:
            if http_err.code == 429:
                print("   [PAUSA 429] Rate limit atingido. Pausa de 15s para esvaziar balde...")
                time.sleep(15)
            elif http_err.code == 400:
                print("   [AVISO 400] Resposta truncada. Ajustando lote e aguardando 5s...")
                time.sleep(5)
            else:
                print(f"   [ERRO HTTP {http_err.code}] {http_err.reason}. Aguardando 8s...")
                time.sleep(8)
            erros += 1

        except Exception as e:
            print(f"   [ERRO] Falha ao processar lote: {e}. Aguardando 4s...")
            time.sleep(4)
            erros += 1

        if erros >= 15:
            print("[ERRO] Mais de 15 erros consecutivos. Pausando minerador.")
            break

if __name__ == "__main__":
    main()
