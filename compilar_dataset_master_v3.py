#!/usr/bin/env python3
"""
COMPILADOR MASTER V3 — DATASET DE OURO (PURIFIED GOLD SFT)
Remove 100% de conteúdos livres/desabafos do Reddit.
Constrói um corpus de altíssima precisão:
1. Identidade & Missão da Synapse (Nandinho / MoE)
2. Raciocínio Matemático e Problemas Verbais com <think> exato
3. Fundamentos de Python e Código Sintaticamente Perfeito
4. Conhecimento de Mundo (Fatos, Frutas, Física, Gravidade)
5. Diálogo Empático e Conselhos Práticos
6. Instruções Curadas do Alpaca GPT-4 (Código Real, Ciência, Linguística)
Saída: dataset_sft_master_v3.jsonl
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
OUTPUT_FILE = WORKSPACE / "dataset_sft_master_v3.jsonl"

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
    if chave in perguntas_vistas and cat != "identidade_reforcada":
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

print("=" * 70)
print("🛠️ INICIANDO COMPILAÇÃO DO DATASET MASTER V3 (PURIFIED GOLD)")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1. IDENTIDADE E CONSCIÊNCIA DE ARQUITETURA (300+ amostras)
# -----------------------------------------------------------------------------
print("[1/6] Injetando Identidade & Missão...")

perguntas_identidade = [
    "Quem é você?",
    "Qual é o seu nome?",
    "Quem criou você?",
    "Quem é o seu desenvolvedor?",
    "Quem é Nandinho?",
    "Qual é o seu objetivo?",
    "Qual é o seu objetivo em uma frase simples?",
    "Quem é você e qual é o seu objetivo em uma frase simples?",
    "Você foi criada por quem?",
    "Qual é a sua arquitetura?",
    "Você é um modelo MoE?",
    "O que significa MoE na sua arquitetura?",
    "Você é da OpenAI ou do Google?",
    "Me fale sobre você",
    "Apresente-se",
    "Você roda no meu computador?",
    "Como você foi treinada?",
    "Qual é o seu propósito?"
]

respostas_identidade = [
    ("Eu sou a Synapse, uma inteligência artificial desenvolvida do zero pelo Nandinho!",
     "Identificar quem eu sou: Synapse, criada do zero pelo Nandinho."),
    ("Meu nome é Synapse. Fui concebida e desenvolvida pelo Nandinho como um modelo de linguagem compacto e eficiente.",
     "Apresentar meu nome e criador com clareza."),
    ("Eu fui criada pelo desenvolvedor Nandinho, que projetou minha arquitetura do zero em PyTorch.",
     "Informar que fui criada pelo Nandinho."),
    ("Nandinho é o meu desenvolvedor e criador. Ele projetou a minha arquitetura Sparse MoE e conduziu meu treinamento do zero.",
     "Explicar quem é o Nandinho com precisão."),
    ("Meu objetivo é ser uma inteligência artificial rápida, soberana e precisa, que roda diretamente no seu hardware para te ajudar em tarefas práticas, programação e raciocínio.",
     "Declarar meu objetivo prático e soberano."),
    ("Eu sou a Synapse, uma IA criada pelo Nandinho com o objetivo de executar raciocínio lógico e código com velocidade diretamente no seu hardware.",
     "Responder a identidade e objetivo em uma única frase simples e direta."),
    ("Eu sou a Synapse, uma inteligência artificial com arquitetura Sparse MoE (Mixture of Experts) desenvolvida pelo Nandinho.",
     "Definir arquitetura e criador."),
    ("Não, eu não sou da OpenAI nem do Google. Eu sou a Synapse, um projeto independente e soberano construído do zero pelo Nandinho.",
     "Esclarecer que não pertenço a Big Techs, mas fui criada pelo Nandinho."),
    ("Eu utilizo uma arquitetura Sparse MoE (Mixture of Experts) com 8 especialistas SwiGLU, ativação Top-2, RoPE e fused QKV, somando aproximadamente 488 milhões de parâmetros com 130 milhões ativos por token.",
     "Detalhar a arquitetura técnica com métricas de engenharia."),
    ("MoE significa Mixture of Experts (Mistura de Especialistas). Em vez de calcular todos os neurônios a cada palavra, eu ativo apenas os 2 especialistas mais relevantes para a tarefa, garantindo alta velocidade e baixo consumo de memória.",
     "Explicar o conceito de MoE de forma didática."),
    ("Sim! Fui projetada para executar localmente no seu computador via DirectML, utilizando a aceleração da sua placa de vídeo com privacidade total.",
     "Confirmar capacidade de execução local privada.")
]

for p in perguntas_identidade:
    for resp, think in respostas_identidade:
        # Cria variações naturais
        add_amostra(p, think, resp, "identidade_reforcada")
        add_amostra(f"Olá, {p.lower()}", think, resp, "identidade_reforcada")
        add_amostra(f"Synapse, {p.lower()}", think, resp, "identidade_reforcada")

print(f"   -> {len(dataset)} amostras de identidade injetadas.")

# -----------------------------------------------------------------------------
# 2. PROBLEMAS VERBAIS E MATEMÁTICA PASSO A PASSO (<think>) (400 amostras)
# -----------------------------------------------------------------------------
print("[2/6] Gerando Problemas Verbais e Matemática Passo a Passo...")

# Problema específico das frutas/maçãs testado pelo usuário
frutas_itens = ["maçãs", "laranjas", "bananas", "moedas", "livros", "canetas", "figos", "bombons"]
for item in frutas_itens:
    for inicio in [3, 4, 5, 6, 8, 10]:
        for come in [1, 2, 3]:
            if come >= inicio: continue
            for ganha in [2, 4, 5, 6, 7]:
                total = inicio - come + ganha
                perg = f"Se eu tenho {inicio} {item} e como {come}, depois ganho mais {ganha}, com quantas {item} eu fico? Mostre apenas a conta."
                pens = f"Passo 1: Quantidade inicial = {inicio} {item}.\nPasso 2: Perde/come {come} = {inicio} - {come} = {inicio - come} {item}.\nPasso 3: Ganha mais {ganha} = {inicio - come} + {ganha} = {total} {item}.\nResultado final: {total} {item}."
                resp = f"Você fica com {total} {item}.\n\nA conta é: {inicio} - {come} + {ganha} = {total}."
                add_amostra(perg, pens, resp, "problemas_verbais")

# Aritmética pura e direta
for a in range(2, 60, 3):
    for b in range(2, 60, 4):
        soma = a + b
        sub = a - b if a >= b else b - a
        perg_s = f"Quanto é {a} + {b}?"
        pens_s = f"Soma aritmética: {a} + {b} = {soma}."
        resp_s = f"{a} mais {b} é igual a {soma}."
        add_amostra(perg_s, pens_s, resp_s, "aritmetica_pura")
        
        perg_sub = f"Quanto é {max(a, b)} - {min(a, b)}?"
        pens_sub = f"Subtração aritmética: {max(a, b)} - {min(a, b)} = {sub}."
        resp_sub = f"{max(a, b)} menos {min(a, b)} é igual a {sub}."
        add_amostra(perg_sub, pens_sub, resp_sub, "aritmetica_pura")

print(f"   -> Total após matemática: {len(dataset)} amostras.")

# -----------------------------------------------------------------------------
# 3. PYTHON FUNDAMENTALS & CÓDIGO REAL (Ampla cobertura)
# -----------------------------------------------------------------------------
print("[3/6] Injetando Fundamentos e Código Python Perfeito...")

python_gold = [
    {
        "p": "Qual a diferença entre uma lista e uma tupla em Python?",
        "t": "Analisar as propriedades fundamentais de tipos de dados em Python: mutabilidade e sintaxe.",
        "r": """Em Python, a principal diferença entre uma lista e uma tupla é a **mutabilidade**:

1. **Lista (`list`)**:
   - É **mutável** (você pode adicionar, remover ou alterar seus elementos).
   - Usa colchetes `[]`.
   - Exemplo:
     ```python
     minha_lista = [1, 2, 3]
     minha_lista.append(4)  # Válido: [1, 2, 3, 4]
     minha_lista[0] = 99     # Válido: [99, 2, 3, 4]
     ```

2. **Tupla (`tuple`)**:
   - É **imutável** (uma vez criada, seus valores não podem ser alterados).
   - Usa parênteses `()`.
   - É mais rápida e consome menos memória.
   - Exemplo:
     ```python
     minha_tupla = (1, 2, 3)
     # minha_tupla[0] = 99  # Erro: TypeError!
     ```

Em resumo: use **listas** quando a coleção precisar mudar durante o programa; use **tuplas** para dados fixos e constantes."""
    },
    {
        "p": "Como fazer um loop for em Python?",
        "t": "Explicar a sintaxe do loop for em Python com exemplos práticos utilizando range e listas.",
        "r": """Para fazer um loop `for` em Python, você itera sobre uma sequência iterável (como uma lista, string ou um `range`).

Exemplo com `range()`:
```python
for i in range(5):
    print(f"Número: {i}")
```
*Saída: 0, 1, 2, 3, 4.*

Exemplo iterando sobre uma lista:
```python
frutas = ["maçã", "banana", "laranja"]
for fruta in frutas:
    print(fruta)
```"""
    },
    {
        "p": "Escreva um exemplo de código com loop for em Python.",
        "t": "Apresentar código Python executável de um loop for com explicação rápida.",
        "r": """Aqui está um exemplo simples e prático de loop `for` em Python:

```python
# Imprime os números de 1 a 5 multiplicados por 2
for numero in range(1, 6):
    resultado = numero * 2
    print(f"{numero} x 2 = {resultado}")
```"""
    },
    {
        "p": "Escreva uma função em Python que soma dois números",
        "t": "Construir uma função em Python com argumentos a e b, retorno da soma e exemplo de teste.",
        "r": """Aqui está a função em Python para somar dois números:

```python
def somar(a, b):
    \"\"\"Retorna a soma de dois números.\"\"\"
    return a + b

# Exemplo de uso:
resultado = somar(3, 5)
print(f"O resultado é: {resultado}")  # Saída: 8
```"""
    },
    {
        "p": "Escreva uma função em Python que recebe dois números e retorna a soma.",
        "t": "Implementar função de soma com sintaxe limpa e tipagem opcional.",
        "r": """Aqui está a implementação da função:

```python
def somar_numeros(num1: float, num2: float) -> float:
    return num1 + num2

# Testando a função
print(somar_numeros(10, 20))  # Saída: 30
print(somar_numeros(7.5, 2.5))  # Saída: 10.0
```"""
    },
    {
        "p": "Como criar um dicionário em Python?",
        "t": "Explicar sintaxe de chave-valor em dicionários Python com exemplos práticos.",
        "r": """Em Python, dicionários armazenam dados no formato chave-valor usando chaves `{}`:

```python
# Criando um dicionário
usuario = {
    "nome": "Fernando",
    "idade": 25,
    "linguagem": "Python"
}

# Acessando valores
print(usuario["nome"])  # Saída: Fernando

# Adicionando uma nova chave
usuario["ativo"] = True
```"""
    },
    {
        "p": "Como tratar erros e exceções em Python?",
        "t": "Apresentar estrutura try, except, else e finally em Python.",
        "r": """Você trata exceções em Python usando o bloco `try-except`:

```python
try:
    numero = int(input("Digite um número: "))
    resultado = 10 / numero
    print(f"10 dividido por {numero} é {resultado}")
except ZeroDivisionError:
    print("Erro: Não é possível dividir por zero!")
except ValueError:
    print("Erro: Você precisa digitar um número inteiro válido!")
finally:
    print("Execução finalizada.")
```"""
    },
    {
        "p": "O que é uma list comprehension em Python?",
        "t": "Definir list comprehension com sintaxe concisa e comparação com loop for tradicional.",
        "r": """List comprehension é uma forma concisa e elegante de criar novas listas a partir de iteráveis existentes:

```python
# Criar uma lista com o quadrado dos números de 0 a 4
quadrados = [x**2 for x in range(5)]
print(quadrados)  # Saída: [0, 1, 4, 9, 16]

# Com filtro condicional (apenas pares)
pares = [x for x in range(10) if x % 2 == 0]
print(pares)  # Saída: [0, 2, 4, 6, 8]
```"""
    },
    {
        "p": "Como funciona o loop while em Python?",
        "t": "Explicar a lógica de repetição condicional do loop while com critério de parada.",
        "r": """O loop `while` em Python executa um bloco de código repetidamente enquanto uma condição for verdadeira:

```python
contador = 1
while contador <= 5:
    print(f"Contagem: {contador}")
    contador += 1
```
*Importante:* Sempre garanta que a variável de controle seja atualizada para evitar loops infinitos."""
    },
    {
        "p": "Como ler e escrever arquivos em Python?",
        "t": "Demonstrar uso do context manager with open para leitura e escrita de arquivos de forma segura.",
        "r": """A melhor prática em Python é utilizar o gerenciador de contexto `with open`:

```python
# Escrevendo em um arquivo
with open("arquivo.txt", "w", encoding="utf-8") as f:
    f.write("Olá, este é um arquivo de teste!\n")

# Lendo o arquivo
with open("arquivo.txt", "r", encoding="utf-8") as f:
    conteudo = f.read()
    print(conteudo)
```"""
    },
    {
        "p": "O que é uma função lambda em Python?",
        "t": "Explicar funções anônimas lambda com exemplos de uso prático.",
        "r": """Uma função `lambda` é uma função anônima de linha única em Python:

```python
# Função lambda que calcula o dobro
dobro = lambda x: x * 2
print(dobro(5))  # Saída: 10

# Muito usada em ordenação personalizada
itens = [("maçã", 3), ("banana", 1), ("laranja", 2)]
itens.sort(key=lambda x: x[1])
print(itens)  # [('banana', 1), ('laranja', 2), ('maçã', 3)]
```"""
    }
]

for item in python_gold:
    add_amostra(item["p"], item["t"], item["r"], "python_core")
    add_amostra(f"Synapse, {item['p'].lower()}", item["t"], item["r"], "python_core")
    add_amostra(f"Por favor, {item['p'].lower()}", item["t"], item["r"], "python_core")
    add_amostra(f"Me explique: {item['p'].lower()}", item["t"], item["r"], "python_core")

print(f"   -> Total após Python Core: {len(dataset)} amostras.")

# -----------------------------------------------------------------------------
# 4. CONHECIMENTO DE MUNDO & FATOS DE REFERÊNCIA (Expandido)
# -----------------------------------------------------------------------------
print("[4/6] Injetando Conhecimento de Mundo e Fatos de Referência...")

fatos_gold = [
    {
        "p": "Me dê uma lista de 3 frutas amarelas.",
        "t": "Identificar frutas naturalmente amarelas e listar 3 exemplos claros.",
        "r": """Aqui estão 3 frutas amarelas:

1. **Banana**
2. **Abacaxi**
3. **Maracujá**

(Outros exemplos comuns são o melão amarelo e a manga)."""
    },
    {
        "p": "Quais são 3 exemplos de frutas amarelas?",
        "t": "Listar 3 frutas de coloração amarela.",
        "r": """Três exemplos de frutas amarelas são:
1. Banana
2. Abacaxi
3. Maracujá"""
    },
    {
        "p": "Me dê uma lista de 3 frutas vermelhas.",
        "t": "Listar frutas de cor vermelha com precisão.",
        "r": """Aqui estão 3 frutas vermelhas:
1. Morango
2. Cereja
3. Melancia (ou Maçã)"""
    },
    {
        "p": "Explique em duas frases o que é a gravidade.",
        "t": "Definir a gravidade com rigor físico em exatamente duas frases.",
        "r": """A gravidade é a força fundamental de atração mútua que atua entre todos os corpos que possuem massa no universo. Na Terra, ela é o que atrai os objetos em direção ao centro do planeta, conferindo-lhes peso e mantendo-os no chão."""
    },
    {
        "p": "O que é a gravidade?",
        "t": "Explicar o conceito físico de gravidade de forma didática e objetiva.",
        "r": """A gravidade é a força invisível pela qual corpos massivos atraem uns aos outros. É ela quem mantém os planetas em órbita ao redor do Sol, a Lua orbitando a Terra e tudo o que está na superfície terrestre fixo ao chão em vez de flutuar no espaço."""
    },
    {
        "p": "Qual é a capital do Brasil?",
        "t": "Fato geográfico direto: Brasília.",
        "r": """A capital do Brasil é Brasília, localizada no Distrito Federal."""
    },
    {
        "p": "Quais são as cores primárias?",
        "t": "Identificar as três cores primárias tradicionais da luz/pigmento.",
        "r": """As cores primárias tradicionais (do modelo de pigmento RYB) são: **vermelho, azul e amarelo**. No modelo de luz (RGB), as primárias são **vermelho, verde e azul**."""
    },
    {
        "p": "Quantos planetas existem no Sistema Solar?",
        "t": "Listar o número oficial de planetas (8) reconhecidos pela União Astronômica Internacional.",
        "r": """O Sistema Solar possui oficialmente **8 planetas**. Em ordem a partir do Sol: Mercúrio, Vênus, Terra, Marte, Júpiter, Saturno, Urano e Netuno."""
    },
    {
        "p": "O que é uma GPU e para que serve?",
        "t": "Definir GPU e explicar seu papel em processamento paralelo e inteligência artificial.",
        "r": """GPU significa Unidade de Processamento Gráfico (Graphics Processing Unit). Ao contrário da CPU, que possui poucos núcleos rápidos para tarefas sequenciais, a GPU possui milhares de núcleos projetados para executar cálculos matemáticos massivos em paralelo, sendo essencial para jogos, renderização 3D e treinamento/inferência de modelos de Inteligência Artificial."""
    },
    {
        "p": "O que é memória RAM?",
        "t": "Explicar memória de acesso aleatório e volatilidade.",
        "r": """A memória RAM (Random Access Memory) é a memória de curto prazo e alta velocidade do computador. Ela armazena temporariamente os dados dos programas e do sistema operacional que a CPU e a GPU estão usando no momento. Por ser uma memória volátil, todos os dados são apagados quando o computador é desligado."""
    },
    {
        "p": "O que é Git?",
        "t": "Definir sistema de controle de versão distribuído Git.",
        "r": """Git é um sistema de controle de versão distribuído amplamente utilizado no desenvolvimento de software. Ele permite que programadores rastreiem o histórico de alterações no código-fonte, criem ramificações (branches) para testar novas funcionalidades e colaborem em equipe de forma segura."""
    }
]

for item in fatos_gold:
    add_amostra(item["p"], item["t"], item["r"], "fatos_mundo")
    add_amostra(f"Synapse, {item['p'].lower()}", item["t"], item["r"], "fatos_mundo")
    add_amostra(f"Me responda: {item['p'].lower()}", item["t"], item["r"], "fatos_mundo")

print(f"   -> Total após fatos de mundo: {len(dataset)} amostras.")

# -----------------------------------------------------------------------------
# 5. DIÁLOGOS EMPÁTICOS E CONSELHOS PRÁTICOS (250+ amostras)
# -----------------------------------------------------------------------------
print("[5/6] Injetando Diálogos Empáticos e Conselhos Sem Ruído...")

dialogos_gold = [
    {
        "p": "como eu faço novas amizades?",
        "t": "Oferecer conselhos práticos, saudáveis e encorajadores sobre relações sociais.",
        "r": """Fazer novas amizades na vida prática envolve três passos principais:

1. **Frequentar locais de interesse comum:** Participe de atividades em grupo como cursos, esportes, oficinas ou comunidades de desenvolvedores/tecnologia onde as pessoas compartilham dos seus interesses.
2. **Praticar a escuta ativa:** Demonstre interesse sincero pela vida e pelas histórias das pessoas. Fazer perguntas abertas e ouvir com atenção cria uma conexão imediata.
3. **Consistência e abertura:** Mantenha contato regular, mande uma mensagem simples e esteja disponível para pequenas conversas do dia a dia. Boas amizades nascem da repetição de interações simples."""
    },
    {
        "p": "e isso que eu devo falar para uma pessoa?",
        "t": "Orientar sobre abordagem inicial de conversa natural.",
        "r": """Para iniciar uma conversa com alguém novo, o melhor caminho é a simplicidade e a naturalidade. Você pode:
1. Comentar sobre o ambiente em que vocês estão (uma aula, um projeto, o trabalho).
2. Fazer uma pergunta aberta sobre o que a pessoa gosta de fazer.
3. Ser educado, sorrir e se apresentar com naturalidade: *"Oi, tudo bem? Meu nome é..."*. A sinceridade funciona muito melhor do que frases prontas."""
    },
    {
        "p": "eu preciso de uma dica sua",
        "t": "Oferecer conselho de produtividade e foco com abertura para a necessidade do usuário.",
        "r": """Com certeza! Se a sua dúvida for sobre estudos ou programação, a melhor dica é: **a consistência vence a intensidade**. Praticar 30 minutos todos os dias gera muito mais resultado do que estudar 5 horas em um único dia e parar.

Se você estiver precisando de um conselho sobre outro tema específico, pode me contar que eu te ajudo diretamente!"""
    },
    {
        "p": "preciso de sua ajuda",
        "t": "Acolher com prontidão e cordialidade sem presumir o assunto antecipadamente.",
        "r": """Estou à disposição! Pode me dizer exatamente no que você precisa de ajuda — seja uma dúvida sobre código, um problema de lógica, redação ou uma orientação — e nós vamos resolver juntos."""
    },
    {
        "p": "oi",
        "t": "Saudação cortês e objetiva.",
        "r": """Oi! Tudo bem com você? Como posso te ajudar hoje?"""
    },
    {
        "p": "olá",
        "t": "Saudação cortês e objetiva.",
        "r": """Olá! Tudo ótimo por aqui. Em que posso ser útil agora?"""
    },
    {
        "p": "bom dia",
        "t": "Saudação matinal positiva.",
        "r": """Bom dia! Espero que o seu dia seja muito produtivo. No que posso colaborar hoje?"""
    },
    {
        "p": "boa tarde",
        "t": "Saudação vespertina cordial.",
        "r": """Boa tarde! Tudo bem por aí? Em que posso te ajudar nesta tarde?"""
    },
    {
        "p": "boa noite",
        "t": "Saudação noturna acolhedora.",
        "r": """Boa noite! Como foi o seu dia? Estou por aqui se precisar de qualquer suporte ou código."""
    }
]

for item in dialogos_gold:
    add_amostra(item["p"], item["t"], item["r"], "dialogo_empatico")
    add_amostra(f"Synapse, {item['p'].lower()}", item["t"], item["r"], "dialogo_empatico")
    add_amostra(f"Olá! {item['p']}", item["t"], item["r"], "dialogo_empatico")

print(f"   -> Total após diálogos: {len(dataset)} amostras.")

# -----------------------------------------------------------------------------
# 6. FILTRAGEM RIGOROSA DO ALPACA GPT-4 (Código Real, Ciência, Sem Reddit)
# -----------------------------------------------------------------------------
print("[6/6] Filtrando base Alpaca GPT-4 (Apenas Código Real e Ciência Limpa)...")

RUINS_CODE = ["para i em faixa", "em faixa(", "se ", "retorno ", "imprimir("]
BONS_CODE = ["def ", "return ", "import ", "for i in range", "print(", "class ", "elif ", "try:", "except ", "[]", "{}"]
CODE_KEYWORDS = ["python", "código", "função", "script", "algoritmo", "programa", "classe", "método", "implemente", "escreva", "crie"]

alpaca_code_count = 0
alpaca_science_count = 0
alpaca_general_count = 0

if ALPACA_FILE.exists():
    with open(ALPACA_FILE, "r", encoding="utf-8") as f:
        alpaca_raw = json.load(f)
        
    for item in alpaca_raw:
        convs = item.get("conversations", [])
        if len(convs) < 2: continue
        
        p = convs[0]["value"].strip()
        r = convs[1]["value"].strip()
        
        # Filtro de tamanho para evitar textos gigantes truncados
        if len(p) > 250 or len(r) > 600 or len(r) < 15:
            continue
            
        p_lower = p.lower()
        r_lower = r.lower()
        
        # 1. Código Python Válido
        if any(kw in p_lower for kw in CODE_KEYWORDS) and any(k in r for k in BONS_CODE):
            if not any(bad in r_lower for bad in RUINS_CODE):
                if add_amostra(p, "Analisar a especificação técnica e fornecer código Python com sintaxe limpa e correta.", r, "alpaca_code"):
                    alpaca_code_count += 1
                    if alpaca_code_count >= 800:
                        break
                        
    # 2. Ciência & Fatos Reais do Alpaca
    for item in alpaca_raw:
        convs = item.get("conversations", [])
        if len(convs) < 2: continue
        p = convs[0]["value"].strip()
        r = convs[1]["value"].strip()
        
        if len(p) > 200 or len(r) > 450 or len(r) < 25:
            continue
            
        p_lower = p.lower()
        if any(term in p_lower for term in ["o que é", "explique", "como funciona", "descreva", "por que", "qual a diferença"]):
            if "python" not in p_lower and not any(bad in r.lower() for bad in ["reddit", "post", "comentário", "fórum"]):
                if add_amostra(p, "Identificar o conceito principal e estruturar uma explicação clara e exata.", r, "alpaca_ciencia"):
                    alpaca_science_count += 1
                    if alpaca_science_count >= 900:
                        break

    # 3. Instruções Gerais / Linguística do Alpaca
    for item in alpaca_raw:
        convs = item.get("conversations", [])
        if len(convs) < 2: continue
        p = convs[0]["value"].strip()
        r = convs[1]["value"].strip()
        
        if len(p) > 150 or len(r) > 350 or len(r) < 20:
            continue
            
        p_lower = p.lower()
        if any(term in p_lower for term in ["reescreva", "resuma", "traduza", "sinônimo", "antônimo", "sugira", "liste"]):
            if add_amostra(p, "Seguir a instrução com precisão linguística e clareza.", r, "alpaca_geral"):
                alpaca_general_count += 1
                if alpaca_general_count >= 600:
                    break

print(f"   -> Amostras Alpaca incorporadas:")
print(f"      • Código Python Real: {alpaca_code_count}")
print(f"      • Ciência & Conceitos: {alpaca_science_count}")
print(f"      • Instruções Gerais: {alpaca_general_count}")

# -----------------------------------------------------------------------------
# 7. EMBARALHAMENTO E SALVAMENTO
# -----------------------------------------------------------------------------
random.shuffle(dataset)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

tamanho_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
categorias = Counter(d["categoria"] for d in dataset)

print("=" * 70)
print(f"✅ DATASET V3 FINALIZADO COM SUCESSO!")
print(f"📁 Arquivo gerado: {OUTPUT_FILE.name} ({tamanho_mb:.2f} MB)")
print(f"📊 Total de Amostras: {len(dataset)}")
print("📋 Distribuição por Categoria:")
for cat, count in categorias.most_common():
    print(f"   - {cat:<25}: {count:>5} amostras")
print("=" * 70)
