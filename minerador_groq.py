#!/usr/bin/env python3
"""
MINERADOR GROQ — ALTA VAZÃO (30 RPM / 14.400 RPD)
Gera datasets ricos com raciocínio (<think>) em português brasileiro
usando Llama 3.3 70B / Llama 3.1 8B na infraestrutura ultra-rápida da Groq.
Compatível com Termux, Linux, Windows (apenas biblioteca padrão urllib).
"""

import urllib.request
import urllib.error
import json
import time
import random
import os
import sys

# Configurações da API Groq
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Modelo rápido e com alta cota diária
MODEL_NAME = "llama-3.1-8b-instant"  # Alternativa 70B: "llama-3.3-70b-versatile"
OUTPUT_FILE = "dataset_minerado_groq.jsonl"

TEMAS = [
    ("Programação & Python", "Conceitos reais de Python, listas vs tuplas, funções, decorators, classes, boas práticas e depuração de bugs comuns."),
    ("Lógica & Resolução de Problemas", "Enigmas de lógica do dia a dia, tomada de decisão prática, planejamento financeiro e estimativas simples."),
    ("Cotidiano & Relacionamentos", "Dúvidas reais do dia a dia, conselhos empáticos sobre trabalho, rotina, convivência e autocuidado."),
    ("Ciência & Curiosidades", "Explicações científicas claras sobre astronomia, corpo humano, física cotidiana e história brasileira de forma acessível."),
    ("Carreira & Tecnologia", "Transição de carreira para TI, preparação para entrevistas técnicas, rotina de desenvolvedor e portfólio."),
    ("Finanças para Iniciantes", "Reserva de emergência, controle de gastos, planejamento para sair de dívidas e noções de juros compostos.")
]

def gerar_lote_groq(api_key, tema_titulo, tema_desc):
    prompt_sistema = """Você é um sintetizador de dados de alta precisão para treinamento de assistentes inteligentes em português brasileiro.
Sua função é gerar diálogos realistas e úteis contendo o processo de raciocínio prévio dentro da tag <think>...</think>.
Retorne ESTRITAMENTE um array JSON válido contendo objetos."""

    prompt_usuario = f"""Gere exatamente 8 diálogos COMPLETOS e ÚTEIS sobre o tema: {tema_titulo} ({tema_desc}).

Para cada diálogo forneça:
1. "tema": Nome do tema
2. "pergunta": Pergunta ou dúvida realista em português do Brasil (natural, como pessoas reais perguntam na internet).
3. "pensamento": Raciocínio prévio passo a passo planejando a explicação de forma clara, precisa e empática.
4. "resposta": Resposta final completa, bem formatada com listas ou blocos de código pequenos quando necessário.

Retorne SOMENTE o JSON no formato:
[
  {{"tema": "{tema_titulo}", "pergunta": "...", "pensamento": "...", "resposta": "..."}},
  ...
]"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario}
        ],
        "temperature": 0.8,
        "response_format": {"type": "json_object"} if "70b" in MODEL_NAME else None
    }

    # Se não suportar json_object nativo no 8b, o prompt já força o retorno
    if not payload["response_format"]:
        del payload["response_format"]

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GROQ_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "SynapseMiner/1.0"
        }
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        raw_text = res["choices"][0]["message"]["content"]
        
        # Limpar markdown ```json se o modelo colocar
        clean_text = raw_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
        
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict):
            # Se vier {"dialogos": [...]} ou similar
            for k in parsed:
                if isinstance(parsed[k], list):
                    return parsed[k]
            return [parsed]
        return parsed

def main():
    print("=" * 70)
    print("⚡ MINERADOR GROQ — ALTA VAZÃO DE DADOS (SYNAPSE SFT)")
    print(f"📁 Arquivo de Saída: {OUTPUT_FILE}")
    print(f"🤖 Modelo Base: {MODEL_NAME}")
    print("=" * 70)

    api_key = GROQ_API_KEY
    if not api_key:
        api_key = input("Cole sua Chave da API Groq (gsk_...): ").strip()

    if not api_key.startswith("gsk_"):
        print("⚠️ Chave da Groq inválida! Deve começar com 'gsk_'.")
        sys.exit(1)

    total = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            total = sum(1 for line in f if line.strip())

    print(f"📦 Amostras já existentes no arquivo: {total}\n")

    rodada = 1
    erros_consecutivos = 0

    while True:
        tema_titulo, tema_desc = random.choice(TEMAS)
        hora = time.strftime("%H:%M:%S")
        print(f"[{hora}] [Rodada {rodada}] Solicitando 8 amostras de '{tema_titulo}' à Groq...")

        try:
            inicio = time.time()
            lote = gerar_lote_groq(api_key, tema_titulo, tema_desc)
            tempo_req = time.time() - inicio
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

            erros_consecutivos = 0
            print(f"   ✅ +{novos} amostras salvas em {tempo_req:.2f}s! Total acumulado: {total}")
            rodada += 1

            # Cadência de 2.5 segundos: taxa de ~24 RPM (Groq suporta até 30 RPM na conta free)
            time.sleep(2.5)

        except urllib.error.HTTPError as http_err:
            print(f"   ❌ Erro HTTP {http_err.code}: {http_err.reason}")
            if http_err.code == 429:
                print("   ⏳ Rate limit temporário atingido. Aguardando 20 segundos...")
                time.sleep(20)
            else:
                time.sleep(10)
            erros_consecutivos += 1

        except Exception as e:
            print(f"   ⚠️ Erro ao processar lote: {e}. Aguardando 5s...")
            time.sleep(5)
            erros_consecutivos += 1

        if erros_consecutivos >= 10:
            print("❌ Mais de 10 erros consecutivos. Pausando minerador para checagem.")
            break

if __name__ == "__main__":
    main()
