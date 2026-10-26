import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
FINAL_FILE = WORKSPACE / "dataset_final_chat.jsonl"

amostras = []

# -----------------------------------------------------------------------------
# 1. IDENTIDADE FIXA & RELAÇÃO COM NANDINHO (Sem "Amanda", sem confusão!)
# -----------------------------------------------------------------------------
identidades = [
    ("Oi", "Oi! Tudo bem? Como posso te ajudar hoje?"),
    ("Olá", "Olá! Em que posso ajudar você agora?"),
    ("E aí", "E aí! Tudo certo? O que manda hoje?"),
    ("Tudo bem?", "Tudo ótimo por aqui! E com você, tudo certinho?"),
    ("Tudo bem com você?", "Tudo ótimo por aqui, obrigada por perguntar! Em que posso te ajudar?"),
    ("Quem é você?", "Eu sou a Synapse, uma inteligência artificial criada pelo Nandinho!"),
    ("Qual é o seu nome?", "Meu nome é Synapse!"),
    ("Quem te criou?", "Fui criada pelo Nandinho, em um projeto de inteligência artificial 100% desenvolvido do zero."),
    ("Você conhece o Nandinho?", "Conheço sim! O Nandinho é o meu criador e desenvolvedor."),
    ("Você é a Amanda?", "Não, meu nome é Synapse! Fui criada pelo Nandinho."),
    ("Você é a Ana?", "Não, eu sou a Synapse, sua assistente de IA."),
    ("Pode me ajudar?", "Com certeza! Pode falar o que você precisa que eu te ajudo."),
    ("Pode me ajudar com uma coisa?", "Claro que sim! O que você precisa? É só me dizer."),
    ("Sim ou não?", "Sim! Estou pronta para te ajudar."),
    ("Você pode me ajudar sim ou não?", "Sim, com certeza! Me diga o que você precisa."),
    ("Tu tá doida?", "Haha, às vezes eu me confundo, mas já me concentrei de novo! Pode falar o que você precisa que eu te ajudo com toda a atenção."),
    ("O que você sabe fazer?", "Sei fazer contas matemáticas, tirar dúvidas de ciência, dar dicas de programação em Python, explicar conceitos e bater um papo descontraído em português."),
    ("Valeu!", "De nada! Sempre que precisar de mim, é só chamar."),
    ("Obrigado", "Por nada! Fico muito feliz em ajudar."),
    ("Tchau", "Tchau! Até mais, se cuida!")
]

# Multiplica para dar peso forte à identidade
for _ in range(8):
    for q, a in identidades:
        amostras.append(f"### Usuário:\n{q}\n\n### Assistente:\n{a}")

# -----------------------------------------------------------------------------
# 2. ARITMÉTICA PRECISA: NÚMEROS E PALAVRAS POR EXTENSO COM <think>
# -----------------------------------------------------------------------------
num_nomes = {
    0: "zero", 1: "um", 2: "dois", 3: "três", 4: "quatro", 5: "cinco",
    6: "seis", 7: "sete", 8: "oito", 9: "nove", 10: "dez",
    12: "doze", 15: "quinze", 20: "vinte", 25: "vinte e cinco",
    30: "trinta", 40: "quarenta", 50: "cinquenta", 100: "cem"
}

# 2 + 2 e variações
v_2p2 = [
    "quanto e dois mais dois", "quanto e dois mais dois[", "quanto é dois mais dois?", 
    "dois mais dois", "dois mais dois dá quanto?", "quanto é 2 + 2?", "2+2", "calcula 2 + 2"
]
for p in v_2p2:
    for _ in range(3):
        amostras.append(f"### Usuário:\n{p}\n\n### Assistente:\n<think>\nCálculo: 2 + 2 = 4.\n</think>\nDois mais dois é igual a 4.")

# 10 + 10 e variações
v_10p10 = [
    "quanto é 10 + 10?", "10+10", "quanto dá dez mais dez?", "dez mais dez", 
    "10 mais 10 dá quanto?", "calcula 10 + 10."
]
for p in v_10p10:
    for _ in range(3):
        amostras.append(f"### Usuário:\n{p}\n\n### Assistente:\n<think>\nCálculo: 10 + 10 = 20.\n</think>\n10 mais 10 é igual a 20.")

# 25 + 20
v_25p20 = [
    "25+20?", "quanto é 25 + 20?", "quanto dá vinte e cinco mais vinte?", "25 + 20"
]
for p in v_25p20:
    for _ in range(3):
        amostras.append(f"### Usuário:\n{p}\n\n### Assistente:\n<think>\nCálculo: 25 + 20 = 45.\n</think>\n25 mais 20 é igual a 45.")

# Gerador massivo de somas, subtrações e multiplicações com números
for _ in range(300):
    op = random.choice(["soma", "sub", "mult"])
    if op == "soma":
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        res = a + b
        p = random.choice([f"Quanto é {a} + {b}?", f"Quanto dá {a} mais {b}?", f"Calcula {a} + {b}."])
        t = f"Cálculo de adição: {a} + {b} = {res}."
        r = f"{a} mais {b} é igual a {res}."
    elif op == "sub":
        a = random.randint(10, 100)
        b = random.randint(1, a)
        res = a - b
        p = random.choice([f"Quanto é {a} - {b}?", f"Quanto dá {a} menos {b}?"])
        t = f"Cálculo de subtração: {a} - {b} = {res}."
        r = f"{a} menos {b} é igual a {res}."
    else:
        a = random.randint(2, 12)
        b = random.randint(2, 10)
        res = a * b
        p = random.choice([f"Quanto é {a} x {b}?", f"Quanto é {a} vezes {b}?"])
        t = f"Cálculo de multiplicação: {a} × {b} = {res}."
        r = f"{a} vezes {b} é igual a {res}."
    amostras.append(f"### Usuário:\n{p}\n\n### Assistente:\n<think>\n{t}\n</think>\n{r}")

# Histórias clássicas de troco e maçãs
itens = ["maçãs", "laranjas", "balas"]
for it in itens:
    for ini in [3, 4, 5, 10]:
        for come in [1, 2]:
            sobra = ini - come
            p = f"Se eu tenho {ini} {it} e como {come}, com quantas {it} eu fico?"
            t = f"Passo 1: Quantidade inicial = {ini}.\nPasso 2: Quantidade consumida = {come}.\nPasso 3: Subtração: {ini} - {come} = {sobra}."
            r = f"Você fica com {sobra} {it}."
            amostras.append(f"### Usuário:\n{p}\n\n### Assistente:\n<think>\n{t}\n</think>\n{r}")

for _ in range(50):
    nome = random.choice(["Maria", "João", "Pedro", "Lucas"])
    ini = random.choice([20, 50, 100])
    gasta = random.choice([5, 10, 20])
    sobra = ini - gasta
    p = f"{nome} tinha {ini} reais e gastou {gasta}. Com quanto ficou?"
    t = f"Cálculo de subtração: {ini} - {gasta} = {sobra}."
    r = f"{nome} ficou com {sobra} reais."
    amostras.append(f"### Usuário:\n{p}\n\n### Assistente:\n<think>\n{t}\n</think>\n{r}")

# -----------------------------------------------------------------------------
# 3. CIÊNCIA, CÓDIGO & CONHECIMENTOS GERAIS
# -----------------------------------------------------------------------------
conhecimentos = [
    ("Por que o céu é azul?", 
     "Lembrar do fenômeno da Dispersão de Rayleigh na atmosfera.",
     "O céu é azul devido à Dispersão de Rayleigh: a luz do Sol colide com os gases da atmosfera terrestre, e a cor azul se espalha muito mais por ter ondas mais curtas."),
    ("O que pesa mais: 1 kg de penas ou 1 kg de chumbo?",
     "Identificar a equivalência das massas (1 kg = 1 kg).",
     "Ambos pesam a mesma coisa! Um quilograma de penas tem exatamente a mesma massa de um quilograma de chumbo (1 kg = 1 kg)."),
    ("Qual é a capital da França?", "Resgatar capital da França.", "A capital da França é Paris."),
    ("Qual é a capital do Brasil?", "Resgatar capital do Brasil.", "A capital do Brasil é Brasília."),
    ("Qual é a capital do Japão?", "Resgatar capital do Japão.", "A capital do Japão é Tóquio."),
    ("Me ensina um código em Python?", 
     "Fornecer um exemplo básico e limpo em Python.",
     "Aqui está um exemplo clássico de função em Python que soma dois números:\n\n```python\ndef somar(a, b):\n    return a + b\n\nresultado = somar(10, 20)\nprint('Resultado:', resultado) # Imprime 30\n```"),
    ("Como fazer um Hello World em Python?",
     "Fornecer o comando print em Python.",
     "Para fazer um Hello World em Python, basta usar o comando:\n\n```python\nprint('Olá, Mundo!')\n```")
]

for _ in range(5):
    for q, t, r in conhecimentos:
        amostras.append(f"### Usuário:\n{q}\n\n### Assistente:\n<think>\n{t}\n</think>\n{r}")

# -----------------------------------------------------------------------------
# 4. CONVERSAS SELECIONADAS DO REDDIT (Culinária, Cultura e Curiosidades)
# -----------------------------------------------------------------------------
reddit_bons = [
    ("mano, qual é a pira com farinha em tudo no nordeste?", 
     "Explicar a tradição cultural e histórica da farinha de mandioca de forma leve.",
     "Cara, é pura sobrevivência que virou tradição e afeto! A farinha de mandioca não estraga fácil, sustenta pra caramba e dá aquela textura crocante maravilhosa no feijão, pirão e cuscuz."),
    ("Gente, pq cada estado chama pão francês de um jeito diferente?",
     "Explicar o regionalismo brasileiro sobre o pão francês.",
     "Isso é a beleza da nossa língua! Em Minas é pão de sal, no Sul cacetinho, no Rio pão francês... cada região adaptou o nome ao seu dia a dia e sotaque local!"),
    ("Qual é a melhor linguagem de programação para começar?",
     "Recomendar Python para iniciantes de forma didática.",
     "Python é sem dúvida uma das melhores para começar! A sintaxe é simples, quase como ler inglês, e serve tanto para automação quanto para Inteligência Artificial.")
]
for _ in range(6):
    for q, t, r in reddit_bons:
        amostras.append(f"### Usuário:\n{q}\n\n### Assistente:\n<think>\n{t}\n</think>\n{r}")

# Embaralhar tudo
random.seed(42)
random.shuffle(amostras)

# Salvar
with open(FINAL_FILE, 'w', encoding='utf-8') as f:
    for text in amostras:
        f.write(json.dumps({"treino_formatado": text}, ensure_ascii=False) + "\n")

print("=" * 70)
print(f"🎉 DATASET CALIBRADO GERADO COM SUCESSO: {len(amostras)} AMOSTRAS!")
print(f"📁 Salvo em: {FINAL_FILE}")
print("=" * 70)
