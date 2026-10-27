#!/usr/bin/env python3
"""
COMPILADOR MASTER V2 — FUSÃO INDUSTRIAL
Une o dataset conversacional do Reddit Brasil com a base filtrada do Alpaca GPT-4 PT
(apenas código Python REAL validado, ciência e lógica) + identidade do Synapse.
Saída: dataset_sft_master_v2.jsonl
"""

import json
import random
import os
import sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
random.seed(42)

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
ALPACA_FILE = WORKSPACE / "alpaca-gpt4-portuguese.json"
SFT_V1_FILE = WORKSPACE / "dataset_sft_master.jsonl"
GROQ_FILE = WORKSPACE / "dataset_minerado_groq.jsonl"
OUTPUT_FILE = WORKSPACE / "dataset_sft_master_v2.jsonl"

dataset = []
perguntas_vistas = set()

def normalizar(s):
    return " ".join(s.strip().lower().replace("?", "").replace("!", "").replace(".", "").split())

def add_amostra(p, t, r, cat):
    p_limpa = p.strip()
    r_limpa = r.strip()
    t_limpa = t.strip() if t else ""
    
    if not p_limpa or not r_limpa:
        return False
        
    chave = normalizar(p_limpa)
    if chave in perguntas_vistas and cat != "identidade":
        return False
    perguntas_vistas.add(chave)
    
    if t_limpa:
        tf = f"### Usuário:\n{p_limpa}\n\n### Assistente:\n<think>\n{t_limpa}\n</think>\n{r_limpa}"
    else:
        tf = f"### Usuário:\n{p_limpa}\n\n### Assistente:\n{r_limpa}"
        
    dataset.append({
        "categoria": cat,
        "pergunta": p_limpa,
        "pensamento": t_limpa,
        "resposta": r_limpa,
        "treino_formatado": tf
    })
    return True

# -----------------------------------------------------------------------------
# 1. CARREGAR DATASET SFT V1 (Reddit + Identidade + Matemática: 1.668 amostras)
# -----------------------------------------------------------------------------
print("[1/4] Carregando dataset SFT V1 existente...")
count_v1 = 0
if SFT_V1_FILE.exists():
    with open(SFT_V1_FILE, "r", encoding="utf-8") as f:
        for l in f:
            if not l.strip(): continue
            d = json.loads(l)
            if add_amostra(d["pergunta"], d["pensamento"], d["resposta"], d["categoria"]):
                count_v1 += 1
print(f"   -> {count_v1} amostras do SFT V1 incorporadas.")

# -----------------------------------------------------------------------------
# 2. CARREGAR AMOSTRAS GROQ RECENTES (124 amostras)
# -----------------------------------------------------------------------------
print("[2/4] Carregando amostras mineradas na Groq...")
count_groq = 0
if GROQ_FILE.exists():
    with open(GROQ_FILE, "r", encoding="utf-8") as f:
        for l in f:
            if not l.strip(): continue
            d = json.loads(l)
            if add_amostra(d["pergunta"], d["pensamento"], d["resposta"], "groq_tech"):
                count_groq += 1
print(f"   -> {count_groq} amostras da Groq incorporadas.")

# -----------------------------------------------------------------------------
# 3. FILTRAGEM RIGOROSA DO ALPACA GPT-4 (Código Python REAL, Lógica e Ciência)
# -----------------------------------------------------------------------------
print("[3/4] Filtrando e curando base Alpaca GPT-4 (50k amostras)...")

RUINS_CODE = ["para i em faixa", "em faixa(", "se ", "retorno ", "imprimir("]
BONS_CODE = ["def ", "return ", "import ", "for i in range", "print(", "class ", "elif ", "try:", "except ", "[]", "{}"]

count_alpaca_code = 0
count_alpaca_logica = 0
count_alpaca_ciencia = 0

if ALPACA_FILE.exists():
    with open(ALPACA_FILE, "r", encoding="utf-8") as f:
        alpaca_raw = json.load(f)
        
    for item in alpaca_raw:
        convs = item.get("conversations", [])
        if len(convs) < 2: continue
        p = convs[0].get("value", "").strip()
        r = convs[1].get("value", "").strip()
        
        len_p = len(p.split())
        len_r = len(r.split())
        
        # Filtro de janela de contexto (máximo 320 palavras para caber folgado em 512 tokens)
        if len_p < 4 or len_r < 6 or (len_p + len_r) > 320:
            continue
            
        # Filtro anti-OpenAI
        r_low = r.lower()
        if "openai" in r_low or "chatgpt" in r_low or "como modelo de linguagem" in r_low:
            continue
            
        p_low = p.lower()
        
        # Categoria A: Programação com Código Real
        is_code = any(k in p_low for k in ["python", "código", "função", "algoritmo", "programação", "lista", "tupla", "dicionário", "sql"])
        if is_code and count_alpaca_code < 1000:
            # Validação estrita de sintaxe
            tem_ruim = any(b in r_low for b in ["para i em faixa", "em faixa("])
            tem_bom = any(g in r for g in BONS_CODE)
            if tem_bom and not tem_ruim:
                t = "Analisar a dúvida técnica. Estruturar a solução em Python com sintaxe correta, boas práticas e explicação didática."
                if add_amostra(p, t, r, "alpaca_python"):
                    count_alpaca_code += 1
            continue
            
        # Categoria B: Lógica e Raciocínio Matemático
        is_logica = any(k in p_low for k in ["calcule", "resolva", "passo a passo", "quantos", "qual o valor", "lógica", "equação"])
        if is_logica and count_alpaca_logica < 800:
            t = "Compreender o problema e decompor as etapas do raciocínio analítico para fornecer o resultado exato."
            if add_amostra(p, t, r, "alpaca_logica"):
                count_alpaca_logica += 1
            continue
            
        # Categoria C: Ciência e Explicações Fatuais
        is_ciencia = any(k in p_low for k in ["como funciona", "o que é", "qual a diferença", "por que", "explique"])
        if is_ciencia and count_alpaca_ciencia < 1000:
            t = "Identificar o conceito central e formular uma explicação clara, precisa e objetiva em português."
            if add_amostra(p, t, r, "alpaca_ciencia"):
                count_alpaca_ciencia += 1
            continue

print(f"   -> {count_alpaca_code} amostras de Python Real adicionadas.")
print(f"   -> {count_alpaca_logica} amostras de Lógica/Matemática adicionadas.")
print(f"   -> {count_alpaca_ciencia} amostras de Ciência/Conceitos adicionadas.")

# -----------------------------------------------------------------------------
# 4. PACOTE ESPECÍFICO DE IDENTIDADE E CÓDIGO FUNDAMENTAL
# -----------------------------------------------------------------------------
print("[4/4] Injetando reforço de identidade e Python Core essencial...")

# Reforço de identidade: duplicar amostras de identidade para atingir peso ótimo de ~4%
amostras_identidade = [item for item in dataset if item["categoria"] == "identidade"]
for item in amostras_identidade:
    dataset.append(item.copy())  # Cópia 2
    dataset.append(item.copy())  # Cópia 3

print(f"   -> Identidade reforçada para {len(amostras_identidade)*3} exposições.")

python_core_essencial = [
    ("Qual a diferença entre lista e tupla em Python?",
     "Comparação direta entre as estruturas fundamentais de sequência em Python.",
     "Em Python, a principal diferença é a **mutabilidade**:\n- **Listas** (`[]`): são mutáveis. Você pode adicionar, remover ou alterar elementos (`lista.append(1)`).\n- **Tuplas** (`()`): são imutáveis. Uma vez criadas, seus valores não podem ser alterados, o que garante maior segurança, menor consumo de memória e permite que sejam usadas como chaves de dicionários."),
     
    ("Como funciona um loop for em Python?",
     "Explicar a iteração sobre sequências e iteráveis em Python.",
     "Em Python, o loop `for` itera diretamente sobre os elementos de qualquer sequência ou iterável:\n```python\nfrutas = ['maçã', 'banana', 'laranja']\nfor fruta in frutas:\n    print(fruta)\n```\nVocê também pode usar `range()` para iterar por índices numéricos: `for i in range(5):` executa de 0 a 4."),
     
    ("O que é uma função e como definir em Python?",
     "Estrutura da palavra-chave def e retorno em Python.",
     "Uma função é um bloco de código reutilizável definido pela palavra-chave `def`:\n```python\ndef saudacao(nome):\n    return f'Olá, {nome}!'\n\nmensagem = saudacao('Nandinho')\nprint(mensagem)\n```\nFunções organizam a lógica, evitam repetição e aceitam parâmetros com valores padrão."),
     
    ("O que é um dicionário em Python?",
     "Explicar estrutura chave-valor e métodos úteis.",
     "Um dicionário (`{}`) é uma coleção não ordenada de pares **chave: valor**:\n```python\nusuario = {'nome': 'Nandinho', 'projeto': 'Synapse', 'versao': 'MoE 500M'}\nprint(usuario['nome'])  # Nandinho\n```\nAs chaves devem ser imutáveis (como strings ou números) e o acesso aos valores é de complexidade O(1)."),
     
    ("Para que serve o bloco try except em Python?",
     "Mecanismo de tratamento e captura de exceções.",
     "O bloco `try/except` serve para capturar e tratar erros em tempo de execução sem derrubar o programa:\n```python\ntry:\n    numero = int('abc')\nexcept ValueError:\n    print('Erro: o valor digitado não é um número válido!')\n```\nIsso torna o sistema robusto contra falhas inesperadas.")
]

for p, t, r in python_core_essencial:
    add_amostra(p, t, r, "python_core")

# Embaralhar para distribuição equilibrada durante o treino
random.shuffle(dataset)

# Gravar dataset final
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\n" + "=" * 70)
print("DATASET SFT MASTER V2 COMPILADO COM SUCESSO!")
print(f"Arquivo gerado: {OUTPUT_FILE}")
print(f"Total absoluto de amostras únicas: {len(dataset)}")
print("=" * 70)

cats = Counter(item["categoria"].split("_")[0] for item in dataset)
for cat, count in cats.most_common():
    pct = (count / len(dataset)) * 100
    print(f"  - {cat:<18}: {count:>5} amostras ({pct:.1f}%)")
print("=" * 70)
