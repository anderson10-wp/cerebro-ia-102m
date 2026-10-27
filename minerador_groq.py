#!/usr/bin/env python3
"""
MINERADOR GROQ TURBO (OPENAI/GPT-OSS-20B)
Alta vazão: 1.300+ tokens a cada 1.8 segundos com modo JSON estrito!
Gera diálogos estruturados com raciocínio (<think>) em português brasileiro.
"""

import urllib.request
import urllib.error
import json
import time
import random
import os
import sys

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
    
    print("\n🔑 Nenhuma chave Groq encontrada.")
    k = input("Cole sua Chave da API Groq (gsk_...): ").strip()
    if k.startswith("gsk_"):
        try:
            with open(".groq_key", "w", encoding="utf-8") as f:
                f.write(k)
            print("💾 Chave salva em .groq_key para próximas execuções!\n")
        except Exception:
            pass
        return k
    print("❌ Chave inválida! Deve começar com 'gsk_'.")
    sys.exit(1)

def gerar_lote_groq(api_key, tema_titulo, tema_desc):
    prompt_sistema = """Você é um sintetizador de dados para treinamento de inteligência artificial em português do Brasil.
Retorne SEMPRE e ESTRITAMENTE um objeto JSON com a chave "dialogos" contendo uma lista de 6 objetos.
Cada objeto DEVE ter os campos:
- "tema": título do tema
- "pergunta": pergunta realista e natural do usuário em português
- "pensamento": reflexão no modo pensar (<think>) planejando a resposta de forma clara, técnica e empática
- "resposta": resposta final completa e bem estruturada."""

    prompt_usuario = f"""Gere 6 diálogos variados sobre: {tema_titulo} ({tema_desc}).
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
        "temperature": 0.75,
        "response_format": {"type": "json_object"}
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
        res = json.loads(resp.read().decode("utf-8"))
        raw_text = res["choices"][0]["message"]["content"]
        parsed = json.loads(raw_text)
        
        if isinstance(parsed, dict):
            if "dialogos" in parsed and isinstance(parsed["dialogos"], list):
                return parsed["dialogos"]
            for v in parsed.values():
                if isinstance(v, list):
                    return v
        if isinstance(parsed, list):
            return parsed
        return []

def main():
    print("=" * 70)
    print("⚡ MINERADOR GROQ TURBO — ATIVO (ALTA VELOCIDADE)")
    print(f"📁 Arquivo de Saída: {OUTPUT_FILE}")
    print(f"🤖 Modelo Base: {MODEL_NAME} (~700 tokens/s)")
    print("=" * 70)

    api_key = obter_chave()
    total = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            total = sum(1 for line in f if line.strip())

    print(f"📦 Amostras já existentes no arquivo: {total}\n")

    rodada = 1
    erros = 0

    while True:
        tema_titulo, tema_desc = random.choice(TEMAS)
        hora = time.strftime("%H:%M:%S")
        print(f"[{hora}] [Rodada {rodada}] Minerando 6 amostras de '{tema_titulo}'...")

        try:
            t0 = time.time()
            lote = gerar_lote_groq(api_key, tema_titulo, tema_desc)
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
            print(f"   ✅ +{novos} amostras salvas em {dt:.2f}s! Total acumulado: {total}")
            rodada += 1

            # Intervalo suave de 2.0s
            time.sleep(2.0)

        except urllib.error.HTTPError as http_err:
            print(f"   ❌ Erro HTTP {http_err.code}: {http_err.reason}")
            if http_err.code == 429:
                print("   ⏳ Rate limit atingido. Aguardando 25s...")
                time.sleep(25)
            else:
                time.sleep(10)
            erros += 1

        except Exception as e:
            print(f"   ⚠️ Erro ao processar lote: {e}. Aguardando 3s...")
            time.sleep(3)
            erros += 1

        if erros >= 10:
            print("❌ Mais de 10 erros consecutivos. Pausando minerador.")
            break

if __name__ == "__main__":
    main()
