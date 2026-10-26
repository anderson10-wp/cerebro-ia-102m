#!/usr/bin/env python3
"""
🚀 MINERADOR REDDIT BRASIL PARA O CELULAR (GALAXY A14 / TERMUX)
Gera conversas autênticas do Reddit Brasil usando a API do Gemini direto no celular.
"""

import urllib.request
import urllib.error
import json
import time
import random
import os
import sys

import base64

# Chave da API Gemini
DEFAULT_B64 = "QVEuQWI4Uk42SnBZVmtWN1BOSWJmcTZRRnBtNElQaDVuc1lvdG1DNjhWdzlKVnV2NVBkR1E="
API_KEY = os.environ.get("GEMINI_API_KEY") or base64.b64decode(DEFAULT_B64).decode()
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={API_KEY}"

OUTPUT_FILE = "dataset_minerado_celular.jsonl"

TEMAS = [
    ("r/conversas", "Curiosidades sobre o cotidiano, hobbies incomuns, memórias de infância, histórias engraçadas e reflexões leves sobre a vida adulta."),
    ("r/PergunteReddit", "Perguntas criativas e inusitadas, dilemas do dia a dia, conselhos práticos e histórias surpreendentes."),
    ("r/brdev", "Dúvidas reais de programação, transição de carreira para tecnologia, Python, inteligência artificial e rotina de home office."),
    ("r/brasil", "Culinária típica de cada estado, gírias regionais, costumes brasileiros, cidades históricas e cultura pop nacional."),
    ("r/desabafos", "Conselhos empáticos e acolhedores sobre cansaço da rotina, superação de obstáculos, autocuidado e amizades verdadeiras."),
    ("r/investimentos", "Dicas práticas de organização financeira para iniciantes, como montar reserva de emergência e poupar dinheiro de forma simples."),
    ("r/gamesEcultura", "Discussões sobre jogos favoritos, nostalgia de consoles antigos, trilhas sonoras marcantes e recomendações."),
    ("r/livros", "Recomendações de livros marcantes, hábitos de leitura, autores brasileiros e debates sobre literatura de forma descontraída.")
]

def gerar_lote_reddit(sub, tema):
    prompt = f"""Você é um minerador de conversas autênticas das comunidades brasileiras do Reddit ({sub}).
Tema do lote: {tema}

Gere 4 diálogos completos e variados no estilo real do Reddit brasileiro.
As perguntas dos usuários devem soar naturais, com linguagem do cotidiano da internet brasileira (gírias leves, sem formalismo engessado).

Para cada diálogo:
- "subreddit": "{sub}"
- "pergunta": a dúvida ou reflexão do usuário (como um post no Reddit)
- "pensamento": uma reflexão estratégica dentro do modo pensar (<think>) planejando como responder com empatia, inteligência e tom caloroso
- "resposta": a resposta final do assistente, sendo amigável, informativa e humana

Retorne SOMENTE um JSON válido com a lista de 4 objetos:
[
  {{"subreddit": "...", "pergunta": "...", "pensamento": "...", "resposta": "..."}},
  ...
]"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.88,
            "responseMimeType": "application/json"
        }
    }
    
    req = urllib.request.Request(
        GEMINI_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        raw = res["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(raw)

def main():
    print("=" * 65)
    print("📱 MINERADOR REDDIT BRASIL — ATIVO NO CELULAR!")
    print(f"📁 Arquivo de saída: {OUTPUT_FILE}")
    print("🔋 Dica: Mantenha o celular na tomada e a tela desligada.")
    print("=" * 65)
    
    total = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            total = sum(1 for line in f if line.strip())
            
    print(f"📦 Amostras já existentes no arquivo: {total}\n")
    
    rodada = 1
    while True:
        sub, tema = random.choice(TEMAS)
        hora = time.strftime("%H:%M:%S")
        print(f"[{hora}] [Rodada {rodada}] 🔍 Minerando 4 conversas de {sub}...")
        
        try:
            lote = gerar_lote_reddit(sub, tema)
            novos = 0
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for item in lote:
                    p = item.get("pergunta", "").strip()
                    t = item.get("pensamento", "").strip()
                    r = item.get("resposta", "").strip()
                    if p and r:
                        treino = f"### Usuário:\n{p}\n\n### Assistente:\n<think>\n{t}\n</think>\n{r}"
                        registro = {
                            "subreddit": item.get("subreddit", sub),
                            "pergunta": p,
                            "pensamento": t,
                            "resposta": r,
                            "treino_formatado": treino
                        }
                        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
                        novos += 1
                        total += 1
                        
            print(f"   ✅ +{novos} conversas salvas! Total acumulado no celular: {total}")
            rodada += 1
            time.sleep(3) # Intervalo respeitoso entre chamadas
        except Exception as e:
            print(f"   ⚠️ Erro temporário: {e}. Aguardando 10s...")
            time.sleep(10)

if __name__ == "__main__":
    main()
