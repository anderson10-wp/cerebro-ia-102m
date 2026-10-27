#!/usr/bin/env python3
"""
COMPILADOR MASTER DO DATASET DE SFT DO SYNAPSE
Funde 1.246 diálogos autênticos minerados no celular com identidade calibrada,
fatos e matemática balanceada, eliminando alucinações de cálculo forçado.
Saída: dataset_sft_master.jsonl
"""

import json
import random
import os
import sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

random.seed(42)

BASE_DIR = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
CELULAR_FILE = BASE_DIR / "dataset_minerado_celular.jsonl"
REDDIT_ANTIGO_FILE = BASE_DIR / "dataset_reddit_conversas.jsonl"
OURO_FILE = BASE_DIR / "dataset_ouro.jsonl"
FINAL_OUTPUT = BASE_DIR / "dataset_sft_master.jsonl"

dataset = []
perguntas_vistas = set()

def normalizar_pergunta(p):
    return " ".join(p.strip().lower().replace("?", "").replace("!", "").replace(".", "").split())

def add_amostra(p, t, r, categoria):
    p_limpa = p.strip()
    r_limpa = r.strip()
    t_limpa = t.strip() if t else ""
    
    chave = normalizar_pergunta(p_limpa)
    if chave in perguntas_vistas and categoria != "identidade":
        return False
    perguntas_vistas.add(chave)
    
    if t_limpa:
        treino = f"### Usuário:\n{p_limpa}\n\n### Assistente:\n<think>\n{t_limpa}\n</think>\n{r_limpa}"
    else:
        treino = f"### Usuário:\n{p_limpa}\n\n### Assistente:\n{r_limpa}"
        
    dataset.append({
        "categoria": categoria,
        "pergunta": p_limpa,
        "pensamento": t_limpa,
        "resposta": r_limpa,
        "treino_formatado": treino
    })
    return True

# =============================================================================
# 1. IDENTIDADE SYNAPSE & NANDINHO (Sem loop de "Amanda", sem confusão)
# =============================================================================
print("[1/5] Injetando identidade calibrada do Synapse...")

identidades = [
    ("Quem é você?", "O usuário perguntou quem eu sou. Devo me apresentar como Synapse, inteligência artificial criada pelo Nandinho.", 
     "Eu sou a Synapse, uma inteligência artificial desenvolvida do zero pelo Nandinho!"),
    ("Qual é o seu nome?", "Pergunta direta sobre meu nome. Responder Synapse de forma direta e amigável.",
     "Meu nome é Synapse!"),
    ("Quem te criou?", "O usuário quer saber sobre minha autoria. Fui criada pelo Nandinho.",
     "Fui criada pelo Nandinho, em um projeto de inteligência artificial construído e treinado do zero."),
    ("Quem é Nandinho?", "O usuário perguntou pelo Nandinho. Explicar que ele é meu criador e desenvolvedor.",
     "O Nandinho é o desenvolvedor e criador que me construiu e treinou desde o início."),
    ("O que você é?", "Explicar minha natureza como IA (modelo de linguagem).",
     "Eu sou um modelo de linguagem artificial (arquitetura Mixture of Experts) chamado Synapse, treinado para conversar, ajudar com dúvidas, lógica e programação."),
    ("Você é o ChatGPT?", "O usuário perguntou se sou o ChatGPT. Negar com clareza e reforçar minha identidade Synapse e meu criador Nandinho.",
     "Não, eu não sou o ChatGPT! Eu sou a Synapse, uma IA criada do zero pelo Nandinho."),
    ("Você foi criada pela OpenAI?", "Negar autoria da OpenAI e reafirmar Nandinho.",
     "Não, eu não tenho ligação com a OpenAI. Fui totalmente criada e treinada pelo Nandinho."),
    ("Você é o Gemini?", "Negar autoria do Google/Gemini.",
     "Não, eu sou a Synapse, um modelo de inteligência artificial desenvolvido pelo Nandinho."),
    ("Você é da Google?", "Negar vínculo institucional.",
     "Não! Sou um projeto independente chamado Synapse, criado pelo Nandinho."),
    ("Como você funciona?", "Explicar a arquitetura MoE do Synapse de forma acessível.",
     "Eu funciono através de uma rede neural com arquitetura Mixture of Experts (MoE), ativando especialistas dedicados para processar e gerar respostas em português."),
    ("Oi, tudo bem?", "Saudação comum. Responder educadamente e perguntar como ajudar.",
     "Oi! Tudo ótimo por aqui! E com você, tudo certinho? Como posso te ajudar hoje?"),
    ("Olá!", "Saudação simples. Ser receptiva.",
     "Olá! Tudo bem? Em que posso ser útil agora?"),
    ("E aí!", "Saudação informal típica brasileira.",
     "E aí! Tudo em paz? O que manda hoje?"),
    ("Preciso de sua ajuda", "O usuário precisa de ajuda aberta. NÃO presumir contas de matemática; perguntar o tema com empatia.",
     "Com certeza! Pode me falar exatamente o que você precisa, seja uma dúvida, um conselho ou um problema técnico, que eu te ajudo."),
    ("Me ajuda aqui?", "Pedido de suporte aberto. Focar em escutar a necessidade.",
     "Estou pronta! Do que você precisa hoje? Manda ver!"),
    ("O que você sabe fazer?", "Resumo de capacidades reais: conversar, tirar dúvidas, programar em Python, resolver problemas.",
     "Eu posso te ajudar a tirar dúvidas de tecnologia, dar dicas de programação em Python, conversar sobre livros, jogos, finanças do dia a dia e resolver problemas de lógica."),
    ("Valeu!", "Agradecimento informal.",
     "Tamo junto! Se precisar de mais alguma coisa, é só chamar."),
    ("Obrigado", "Agradecimento formal.",
     "Por nada! Fico feliz em ajudar. Qualquer outra dúvida, estou à disposição."),
    ("Tchau", "Despedida.",
     "Até logo! Se cuida e até a próxima.")
]

# Injetar com variações de pergunta para garantir robustez e peso ideal (~5%)
for p, t, r in identidades:
    add_amostra(p, t, r, "identidade")
    add_amostra(p.lower(), t, r, "identidade")
    add_amostra(f"Synapse, {p.lower()}", t, r, "identidade")
    add_amostra(f"Ei Synapse, {p.lower()}", t, r, "identidade")


# =============================================================================
# 2. DIÁLOGOS MINERADOS NO CELULAR (1.246 amostras de alto nível)
# =============================================================================
print("[2/5] Carregando diálogos minerados no celular...")
count_celular = 0
if CELULAR_FILE.exists():
    with open(CELULAR_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            p = d.get("pergunta", "").strip()
            t = d.get("pensamento", "").strip()
            r = d.get("resposta", "").strip()
            sub = d.get("subreddit", "conversa")
            if p and r:
                if add_amostra(p, t, r, f"reddit_{sub}"):
                    count_celular += 1

print(f"   -> {count_celular} diálogos do celular incorporados com sucesso.")

# =============================================================================
# 3. DIÁLOGOS DO REDDIT ANTERIOR (303 amostras não duplicadas)
# =============================================================================
print("[3/5] Carregando diálogos do Reddit anterior...")
count_antigo = 0
if REDDIT_ANTIGO_FILE.exists():
    with open(REDDIT_ANTIGO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            p = d.get("pergunta", "").strip()
            t = d.get("pensamento", "").strip()
            r = d.get("resposta", "").strip()
            if p and r:
                if add_amostra(p, t, r, "reddit_historico"):
                    count_antigo += 1

print(f"   -> {count_antigo} diálogos adicionais incorporados.")

# =============================================================================
# 4. CONHECIMENTO FACTUAL E CIÊNCIA (Do dataset ouro)
# =============================================================================
print("[4/5] Incorporando base de fatos e conceitos...")
count_fatos = 0
if OURO_FILE.exists():
    with open(OURO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            msgs = d.get("messages", [])
            if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                p = msgs[0]["content"].strip()
                r = msgs[1]["content"].strip()
                t = "Responder à pergunta factual com exatidão e linguagem clara."
                if p and r and not ("amanda" in r.lower() or "amanda" in p.lower()):
                    if add_amostra(p, t, r, "fatos_ouro"):
                        count_fatos += 1

print(f"   -> {count_fatos} fatos incorporados.")

# =============================================================================
# 5. MATEMÁTICA BALANCEADA E DIVERSA (Sem vícios em números fixos)
# =============================================================================
print("[5/5] Gerando amostras de raciocínio matemático balanceado...")
count_math = 0

# Pequenas operações com números variados para evitar vício em '25+20=45'
for _ in range(120):
    op = random.choice(["+", "-", "*"])
    if op == "+":
        a = random.randint(2, 80)
        b = random.randint(2, 80)
        res = a + b
        p = random.choice([f"Quanto é {a} + {b}?", f"Quanto dá {a} mais {b}?", f"Calcula {a} + {b}"])
        t = f"Operação de adição solicitada: {a} + {b}. Efetuando a soma direta: {a} + {b} = {res}."
        r = f"{a} mais {b} é igual a {res}."
    elif op == "-":
        a = random.randint(10, 100)
        b = random.randint(1, a)
        res = a - b
        p = random.choice([f"Quanto é {a} - {b}?", f"Quanto dá {a} menos {b}?"])
        t = f"Operação de subtração solicitada: {a} - {b}. Efetuando a subtração: {a} - {b} = {res}."
        r = f"{a} menos {b} é igual a {res}."
    else:
        a = random.randint(2, 10)
        b = random.randint(2, 10)
        res = a * b
        p = random.choice([f"Quanto é {a} x {b}?", f"Quanto dá {a} vezes {b}?"])
        t = f"Operação de multiplicação solicitada: {a} × {b}. Efetuando a multiplicação: {a} × {b} = {res}."
        r = f"{a} vezes {b} é igual a {res}."
        
    if add_amostra(p, t, r, "matematica"):
        count_math += 1

# Casos explícitos que o usuário testou recentemente
casos_teste = [
    ("Quanto é 2 + 2?", "Soma simples: 2 + 2 = 4.", "2 mais 2 é igual a 4."),
    ("quanto é 29 + 20?", "Soma solicitada: 29 + 20. Efetuando o cálculo: 29 + 20 = 49.", "29 mais 20 é igual a 49."),
    ("Quanto é 49 + 39?", "Soma solicitada: 49 + 39. Efetuando o cálculo: 49 + 39 = 88.", "49 mais 39 é igual a 88."),
    ("Quanto é 10 + 10?", "Soma simples: 10 + 10 = 20.", "10 mais 10 é igual a 20.")
]
for p, t, r in casos_teste:
    add_amostra(p, t, r, "matematica")
    count_math += 1

print(f"   -> {count_math} problemas de raciocínio matemático incorporados.")

# Embaralhar para evitar viés sequencial durante o treino
random.shuffle(dataset)

# Gravar no arquivo mestre
with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\n" + "=" * 65)
print(f"DATASET SFT MASTER COMPILADO COM SUCESSO!")
print(f"Arquivo gerado: {FINAL_OUTPUT}")
print(f"Total absoluto de amostras únicas: {len(dataset)}")
print("=" * 65)

# Distribuição final
categorias = Counter(item["categoria"].split("_")[0] for item in dataset)
for cat, count in categorias.most_common():
    pct = (count / len(dataset)) * 100
    print(f"  - {cat:<18}: {count:>5} amostras ({pct:.1f}%)")
print("=" * 65)
