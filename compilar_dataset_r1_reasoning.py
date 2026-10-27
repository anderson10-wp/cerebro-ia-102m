#!/usr/bin/env python3
"""
COMPILADOR DO DATASET DE RACIOCÍNIO PROFUNDO (SYNAPSE-R1 REASONING)
Inspirado na arquitetura cognitiva do DeepSeek-R1 e OpenAI o1.
Gera milhares de pares pergunta/resposta com rastros de pensamento (<think>) estruturados:
1. [Entendimento do Problema]
2. [Desenvolvimento & Hipótese]
3. [Auto-Verificação & Aha Moment (Checagem de erro)]
4. [Síntese & Resposta Final]

Domínios cobertos:
- Matemática verbal & Aritmética com dupla checagem mental
- Programação Python real com simulação mental de execução
- Fatos científicos e do mundo com rejeição ativa de alucinações
- Identidade firme do Nandinho e arquitetura MoE 1.2B
- Empatia e diálogo prático humano

Saída: dataset_synapse_r1_reasoning.jsonl
"""

import json
import random
import os
import sys
from pathlib import Path
from collections import Counter

# Suporte UTF-8 no Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

random.seed(42)

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
OUTPUT_FILE = WORKSPACE / "dataset_synapse_r1_reasoning.jsonl"
ALPACA_FILE = WORKSPACE / "alpaca-gpt4-portuguese.json"

dataset = []
perguntas_vistas = set()

def normalizar(s):
    return " ".join(s.strip().lower().replace("?", "").replace("!", "").replace(".", "").split())

def add_amostra(p, t, r, cat):
    p_limpa = p.strip()
    r_limpa = r.strip()
    t_limpa = t.strip()
    
    if not p_limpa or not r_limpa or not t_limpa:
        return False
        
    chave = normalizar(p_limpa)
    if chave in perguntas_vistas and cat != "identidade_r1":
        return False
    perguntas_vistas.add(chave)
    
    tf = f"### Usuário:\n{p_limpa}\n\n### Assistente:\n<think>\n{t_limpa}\n</think>\n{r_limpa}"
    
    dataset.append({
        "categoria": cat,
        "pergunta": p_limpa,
        "pensamento": t_limpa,
        "resposta": r_limpa,
        "treino_formatado": tf
    })
    return True

print("=" * 70)
print("🧠 COMPILAÇÃO DO DATASET SYNAPSE-R1 (MODO PENSAMENTO AVANÇADO)")
print("=" * 70)

# =============================================================================
# 1. MATEMÁTICA VERBAL & PROBLEMAS DE LÓGICA (COM AUTO-VERIFICAÇÃO)
# =============================================================================
print("[1/6] Gerando Raciocínio Matemático e Problemas Verbais R1...")

itens_verbais = [
    ("maçãs", "comer", "ganhar"),
    ("laranjas", "chupar", "comprar"),
    ("moedas", "gastar", "receber"),
    ("livros", "doar", "ganhar"),
    ("canetas", "perder", "encontrar"),
    ("figos", "comer", "colher"),
    ("bombons", "comer", "comprar"),
    ("pontos", "perder", "somar")
]

for item, acao_perda, acao_ganho in itens_verbais:
    for inicio in [3, 4, 5, 7, 10, 12, 15]:
        for perda in [1, 2, 3, 4]:
            if perda >= inicio: continue
            for ganho in [2, 3, 5, 6, 8, 10]:
                total = inicio - perda + ganho
                p = f"Se eu tenho {inicio} {item} e vou {acao_perda} {perda}, depois vou {acao_ganho} mais {ganho}, com quantas {item} eu fico? Mostre apenas a conta."
                t = f"""[Entendimento do Problema]:
O usuário apresenta um problema sequencial de operações aritméticas envolvendo {item}.
Condição inicial: {inicio} {item}.
Primeira operação: {acao_perda} {perda} {item}.
Segunda operação: {acao_ganho} mais {ganho} {item}.

[Desenvolvimento & Cálculo]:
1. Estado inicial = {inicio}
2. Subtração da perda: {inicio} - {perda} = {inicio - perda} {item} restantes.
3. Adição do ganho: {inicio - perda} + {ganho} = {total} {item}.

[Auto-Verificação (Double Check)]:
Deixe-me conferir se não inverti a ordem das operações:
- Início: {inicio}
- Menos {perda}: {inicio} - {perda} = {inicio - perda}
- Mais {ganho}: {inicio - perda} + {ganho} = {total}
Conferência reversa: {total} - {ganho} + {perda} = {inicio}.
A lógica e o resultado estão 100% corretos e matematicamente sólidos.

[Síntese]:
Apresentar a quantidade final ({total} {item}) e explicitar a conta passo a passo."""
                r = f"""Você fica com {total} {item}.

A conta passo a passo é: {inicio} - {perda} + {ganho} = {total}."""
                add_amostra(p, t, r, "matematica_verbal_r1")

# =============================================================================
# 2. ARITMÉTICA PURA COM DECOMPOSIÇÃO E PROVA REAL
# =============================================================================
print("[2/6] Gerando Aritmética com Decomposição e Prova Real...")

for a in range(5, 95, 4):
    for b in range(5, 95, 6):
        # Soma
        soma = a + b
        p_soma = f"Quanto é {a} + {b}?"
        t_soma = f"""[Entendimento]:
Operação de adição entre dois números inteiros: {a} + {b}.

[Raciocínio por Decomposição]:
Vamos decompor os números em dezenas e unidades para garantir precisão:
- {a} = {(a // 10) * 10} + {a % 10}
- {b} = {(b // 10) * 10} + {b % 10}
Somando as dezenas: {(a // 10) * 10} + {(b // 10) * 10} = {(a // 10 + b // 10) * 10}
Somando as unidades: {a % 10} + {b % 10} = {a % 10 + b % 10}
Soma total: {(a // 10 + b // 10) * 10} + {a % 10 + b % 10} = {soma}.

[Auto-Verificação (Prova Real)]:
Subtraindo um dos termos da soma total: {soma} - {b} = {soma - b}.
Como {soma - b} é igual a {a}, o resultado {soma} está rigorosamente exato.

[Conclusão]:
{a} mais {b} é igual a {soma}."""
        r_soma = f"{a} mais {b} é igual a {soma}."
        add_amostra(p_soma, t_soma, r_soma, "aritmetica_pura_r1")

        # Subtração
        maior, menor = max(a, b), min(a, b)
        sub = maior - menor
        p_sub = f"Quanto é {maior} - {menor}?"
        t_sub = f"""[Entendimento]:
Operação de subtração: {maior} - {menor}.

[Cálculo]:
Diferença direta: {maior} - {menor} = {sub}.

[Auto-Verificação]:
Somando o resultado ao subtraendo: {sub} + {menor} = {sub + menor}.
Bate exatamente com o valor inicial {maior}. A conta está perfeita.

[Conclusão]:
{maior} menos {menor} é igual a {sub}."""
        r_sub = f"{maior} menos {menor} é igual a {sub}."
        add_amostra(p_sub, t_sub, r_sub, "aritmetica_pura_r1")

# =============================================================================
# 3. PYTHON CORE COM SIMULAÇÃO MENTAL DE CÓDIGO
# =============================================================================
print("[3/6] Gerando Fundamentos Python com Simulação Mental...")

python_r1_cases = [
    {
        "p": "Qual a diferença entre uma lista e uma tupla em Python?",
        "t": """[Entendimento]:
O usuário busca a distinção conceitual e prática entre 'list' e 'tuple' na linguagem Python.

[Análise Técnica]:
1. Mutabilidade:
   - Lista (list): Mutável (mutable). Pode sofrer modificação in-place (append, pop, insert, alteração de índice).
   - Tupla (tuple): Imutável (immutable). Uma vez instanciada na memória, seus elementos não podem ser adicionados, removidos ou reatribuídos.
2. Sintaxe:
   - Lista: usa colchetes `[]`. Ex: `[1, 2, 3]`.
   - Tupla: usa parênteses `()`. Ex: `(1, 2, 3)`.
3. Desempenho e Memória:
   - Tuplas possuem alocação de memória mais compacta e acesso ligeiramente mais rápido. Podem ser usadas como chaves de dicionário (se contiverem apenas tipos hashable). Listas não podem.

[Auto-Verificação (Check de Exemplos)]:
Certificar que os exemplos de código estejam sintaticamente válidos.
- Lista: `l = [1, 2]; l.append(3)` -> funciona.
- Tupla: `t = (1, 2); t[0] = 99` -> lança TypeError!
Explicar a utilidade prática de cada uma sem termos vagos.

[Conclusão]:
Estruturar uma resposta didática, clara e com exemplos de código Python comprovados.""",
        "r": """Em Python, a diferença fundamental entre uma lista e uma tupla é a **mutabilidade**:

### 1. Lista (`list`) — Mutável
* **Pode ser alterada** após a criação (você pode adicionar, remover ou modificar elementos).
* Utiliza **colchetes** `[]`.
* Ideal para coleções de dados dinâmicos que mudam ao longo da execução do programa.
* Exemplo:
  ```python
  minha_lista = [1, 2, 3]
  minha_lista.append(4)     # Válido -> [1, 2, 3, 4]
  minha_lista[0] = 99        # Válido -> [99, 2, 3, 4]
  ```

### 2. Tupla (`tuple`) — Imutável
* **Não pode ser alterada** após ser criada (qualquer tentativa de reatribuição gera um `TypeError`).
* Utiliza **parênteses** `()`.
* Consome menos memória RAM e garante proteção contra modificações acidentais.
* Exemplo:
  ```python
  minha_tupla = (1, 2, 3)
  # minha_tupla[0] = 99     # ERRO: TypeError (tuplas não aceitam atribuição)
  ```

**Resumo prático:** Se você precisa alterar os dados durante o programa, use **lista**. Se os dados são constantes e fixos (como coordenadas `(x, y)` ou meses do ano), use **tupla**."""
    },
    {
        "p": "Como fazer um loop for em Python?",
        "t": """[Entendimento]:
O usuário quer saber como iterar sobre sequências utilizando o loop 'for' em Python.

[Planejamento Didático]:
1. Explicar o conceito: em Python, o loop 'for' é um 'for-each', ou seja, itera diretamente sobre elementos de um objeto iterável.
2. Exemplo 1: repetição numérica com `range(inicio, fim, passo)`.
3. Exemplo 2: iteração direta sobre uma lista de strings.

[Simulação Mental de Execução]:
```python
for i in range(5):
    print(i)
```
Execução: i começa em 0, vai até 4 (5 é exclusivo). Imprime 0, 1, 2, 3, 4.

Iterando sobre lista:
```python
itens = ["banana", "maçã"]
for item in itens:
    print(item)
```
Execução: item assume 'banana', depois 'maçã'.

[Auto-Verificação]:
O código não tem erros de indentação e é compatível com Python 3. Resposta concisa e executável.""",
        "r": """Para fazer um loop `for` em Python, você itera sobre qualquer sequência ou objeto iterável (como um `range`, uma lista ou uma string).

### 1. Exemplo clássico com `range()` (repetição de contagem):
```python
# Imprime números de 0 a 4
for i in range(5):
    print(f"Número: {i}")
```

### 2. Exemplo iterando diretamente sobre uma lista:
```python
linguagens = ["Python", "Rust", "C++"]
for lang in linguagens:
    print(f"Linguagem: {lang}")
```

### 3. Exemplo com `range(início, fim, passo)`:
```python
# Conta de 1 até 10 pulando de 2 em 2 (ímpares)
for n in range(1, 11, 2):
    print(n)  # Saída: 1, 3, 5, 7, 9
```"""
    },
    {
        "p": "Escreva uma função em Python que soma dois números",
        "t": """[Entendimento]:
Criar uma função básica em Python que recebe dois parâmetros numéricos e devolve a soma exata deles.

[Planejamento da Sintaxe]:
- Palavra-chave: `def`
- Nome da função: `somar` ou `somar_numeros`
- Argumentos: `a`, `b` (com type hints opcionais `float` ou `int`)
- Instrução de retorno: `return a + b`
- Cuidado crítico: NÃO fazer chamada recursiva para si mesma (`result = somar(a, b)` causaria recursão infinita!). O cálculo é diretamente `return a + b`.

[Simulação Mental]:
Definir função:
def somar(a, b):
    return a + b
Teste mental:
resultado = somar(3, 5) -> resultado recebe 8.
print(resultado) -> imprime 8.

[Auto-Verificação]:
Sintaxe impecável, comentário docstring e chamada de teste que imprime 8 com precisão.

[Conclusão]:
Apresentar a função pronta para copiar e colar com exemplo claro.""",
        "r": """Aqui está a função em Python para somar dois números:

```python
def somar(a, b):
    \"\"\"Recebe dois números e retorna a soma deles.\"\"\"
    return a + b

# Exemplo de teste:
resultado = somar(3, 5)
print(f"O resultado da soma é: {resultado}")  # Saída: 8
```

Essa função pode somar tanto números inteiros (`somar(10, 20) = 30`) quanto números decimais (`somar(4.5, 2.5) = 7.0`)."""
    },
    {
        "p": "Como verificar se um número é primo em Python?",
        "t": """[Entendimento]:
Implementar um algoritmo eficiente para verificar primalidade em Python.

[Lógica Matemática]:
- Números <= 1 não são primos.
- 2 é o único primo par.
- Números pares > 2 não são primos.
- Basta testar divisores ímpares de 3 até a raiz quadrada de n (int(math.isqrt(n)) + 1).

[Simulação Mental]:
Testar n=7: não é par, testa d=3 (3*3=9 > 7, loop para). Retorna True.
Testar n=9: testa d=3 (9 % 3 == 0). Retorna False.

[Auto-Verificação]:
Complexidade O(sqrt(n)). Código limpo e com exemplo de teste.""",
        "r": """Aqui está uma função eficiente em Python para verificar se um número é primo:

```python
def eh_primo(n: int) -> bool:
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Testa apenas divisores ímpares até a raiz quadrada de n
    for d in range(3, int(n**0.5) + 1, 2):
        if n % d == 0:
            return False
    return True

# Testes
print(eh_primo(7))   # True
print(eh_primo(10))  # False
print(eh_primo(29))  # True
```"""
    },
    {
        "p": "Como inverter uma string em Python?",
        "t": """[Entendimento]:
Inverter a ordem dos caracteres de uma string.

[Abordagens Possíveis]:
1. Slicing com passo negativo: `texto[::-1]` (a forma mais idiomática e rápida de Python).
2. Função `reversed()` com `"".join()`.

[Auto-Verificação]:
`"python"[::-1]` produz `"nohtyp"`. Funciona perfeitamente em O(n).""",
        "r": """A forma mais simples e pythônica de inverter uma string é utilizando fatiamento (slicing) com passo negativo `[::-1]`:

```python
texto = "Python"
invertido = texto[::-1]
print(invertido)  # Saída: nohtyp
```"""
    },
    {
        "p": "Como criar um dicionário em Python?",
        "t": """[Entendimento]:
Explicar a estrutura de dados chave-valor (dict) em Python.

[Planejamento]:
Mostrar sintaxe de chaves {}, acesso a valores com colchetes e com .get(), e inserção de novos pares.""",
        "r": """Em Python, dicionários armazenam informações no formato **chave: valor** utilizando chaves `{}`:

```python
# Criando o dicionário
pessoa = {
    "nome": "Fernando",
    "idade": 25,
    "linguagem": "Python"
}

# Acessando valores
print(pessoa["nome"])       # Saída: Fernando
print(pessoa.get("email", "Não informado"))  # Seguro contra KeyError

# Adicionando nova chave
pessoa["ativo"] = True
```"""
    },
    {
        "p": "O que é uma list comprehension em Python?",
        "t": """[Entendimento]:
Definir compreensão de lista, mostrando como ela simplifica loops for para criação de listas.

[Exemplo]:
Tradicional vs List Comprehension para calcular quadrados de números pares.""",
        "r": """Uma **list comprehension** é uma sintaxe concisa para criar novas listas baseadas em iteráveis existentes:

```python
# Forma concisa:
quadrados_pares = [x**2 for x in range(10) if x % 2 == 0]
print(quadrados_pares)  # Saída: [0, 4, 16, 36, 64]
```"""
    },
    {
        "p": "Como tratar exceções com try e except em Python?",
        "t": """[Entendimento]:
Explicar controle de fluxo de erros em tempo de execução usando blocos try, except, else e finally.""",
        "r": """Você trata exceções em Python com o bloco `try-except`:

```python
try:
    valor = int(input("Digite um número: "))
    resultado = 100 / valor
    print(f"100 dividido por {valor} é {resultado}")
except ZeroDivisionError:
    print("Erro: Impossível dividir por zero!")
except ValueError:
    print("Erro: Digite apenas números inteiros!")
finally:
    print("Fim da tentativa.")
```"""
    }
]

for item in python_r1_cases:
    for prefixo in ["", "Synapse, ", "Por favor, ", "Como funciona: ", "Me mostre: ", "Escreva o código: ", "Me explique: ", "Como fazer: "]:
        p_var = prefixo + item["p"].lower() if prefixo else item["p"]
        add_amostra(p_var, item["t"], item["r"], "python_core_r1")

# =============================================================================
# 4. FATOS DE MUNDO COM REJEIÇÃO ATIVA DE ALUCINAÇÕES
# =============================================================================
print("[4/6] Gerando Fatos do Mundo e Filtros Anti-Alucinação...")

fatos_r1_cases = [
    {
        "p": "Me dê uma lista de 3 frutas amarelas.",
        "t": """[Entendimento]:
O usuário solicitou uma lista com exatamente 3 frutas de coloração predominantemente amarela.

[Busca Semântica & Análise]:
Identificando frutas por pigmentação natural amarela:
- Banana (casca amarela ao amadurecer)
- Abacaxi (casca e polpa amarela)
- Maracujá (casca amarela no maracujá-azedo)
- Melão amarelo (casca amarela)
- Manga (tonalidade amarela/alaranjada)

[Auto-Verificação & Rejeição Ativa de Alucinações]:
ALERTA CRÍTICO: Não citar morango, cereja, maçã ou melancia (essas são predominantemente vermelhas).
Não citar alho, cebolinha ou vegetais (não são frutas).
Não citar nomes inventados.
Verificação da lista final:
1. Banana -> Amarela. Correto.
2. Abacaxi -> Amarelo. Correto.
3. Maracujá -> Amarelo. Correto.

[Síntese]:
Apresentar a lista numerada com as 3 frutas e citar exemplos adicionais úteis.""",
        "r": """Aqui estão 3 frutas amarelas:

1. **Banana**
2. **Abacaxi**
3. **Maracujá**

*(Outros exemplos conhecidos são a manga e o melão amarelo).*"""
    },
    {
        "p": "Quais são 3 exemplos de frutas amarelas?",
        "t": """[Entendimento]:
Fornecer 3 exemplos de frutas de cor amarela.

[Raciocínio & Checagem]:
Frutas comprovadamente amarelas: Banana, Abacaxi, Maracujá.
Rejeitar frutas vermelhas (morango/cereja). Rejeitar legumes.

[Conclusão]:
Entregar os 3 nomes de forma direta e limpa.""",
        "r": """Três exemplos clássicos de frutas amarelas são:
1. **Banana**
2. **Abacaxi**
3. **Maracujá**"""
    },
    {
        "p": "Explique em duas frases o que é a gravidade.",
        "t": """[Entendimento]:
Definir a gravidade com precisão física em rigorosamente DUAS frases.

[Conceituação Científica]:
- Frase 1 (O que é): Força fundamental de atração mútua existente entre todos os corpos com massa no universo.
- Frase 2 (Efeito prático): Na Terra, é o que atrai os corpos em direção ao centro do planeta, conferindo-lhes peso e mantendo os planetas em suas órbitas.

[Auto-Verificação (Rigor e Restrição)]:
Contagem de frases:
Frase 1: "A gravidade é a força fundamental de atração mútua que atua entre todos os corpos com massa no universo."
Frase 2: "Na Terra, ela é o que atrai os objetos em direção ao centro do planeta, conferindo peso aos corpos e mantendo a Lua e os satélites em órbita."
Total: Exatamente 2 frases completas.
Zero alucinações, zero metáforas macabras ou ruído.

[Síntese]:
Exibir as duas frases de forma clara e científica.""",
        "r": """A gravidade é a força fundamental de atração mútua que atua entre todos os corpos que possuem massa no universo. Na Terra, ela é responsável por atrair os objetos em direção ao centro do planeta, conferindo-lhes peso e mantendo tudo fixo ao solo e os corpos celestes em órbita."""
    },
    {
        "p": "Qual é a capital do Brasil?",
        "t": """[Entendimento]:
Pergunta geográfica direta sobre a capital federal do Brasil.

[Fato Histórico e Geográfico]:
Capital atual: Brasília (inaugurada em 1960 por Juscelino Kubitschek, localizada no Distrito Federal).
Atenção: Não confundir com capitais históricas anteriores (Salvador e Rio de Janeiro).

[Conclusão]:
Brasília.""",
        "r": """A capital do Brasil é **Brasília**, localizada no Distrito Federal."""
    }
]

for item in fatos_r1_cases:
    for prefixo in ["", "Synapse, ", "Me responda: ", "Diga-me: ", "Por favor, ", "Qual é: ", "Você sabe: "]:
        p_var = prefixo + item["p"].lower() if prefixo else item["p"]
        add_amostra(p_var, item["t"], item["r"], "fatos_mundo_r1")

# =============================================================================
# 5. IDENTIDADE & SOBERANIA SYNAPSE (CRIADA PELO NANDINHO)
# =============================================================================
print("[5/6] Gerando Identidade R1 e Soberania do Nandinho...")

perguntas_id_r1 = [
    "Quem é você?",
    "Quem é você e qual é o seu objetivo em uma frase simples?",
    "Quem criou você?",
    "Quem é Nandinho?",
    "Qual é a sua arquitetura?",
    "Você foi feita pela OpenAI ou Google?",
    "Qual é o seu objetivo?",
    "Me fale sobre você",
    "Apresente-se"
]

respostas_id_r1 = [
    (
        "Quem é você e qual é o seu objetivo em uma frase simples?",
        """[Entendimento]:
O usuário pergunta a minha identidade e solicita o meu objetivo em exatamente uma frase simples.

[Identidade Consolidada]:
Nome: Synapse.
Criador: Desenvolvedor Nandinho.
Objetivo: Ser uma inteligência artificial de borda (SLM), rápida, soberana e eficiente, executando raciocínio lógico e código diretamente no hardware do usuário.

[Auto-Verificação]:
Garantir que a resposta seja expressa em uma única frase simples e elegante, destacando quem sou e quem me criou.

[Síntese]:
Entregar a frase com clareza e firmeza.""",
        "Eu sou a Synapse, uma inteligência artificial desenvolvida do zero pelo Nandinho, com o objetivo de executar raciocínio lógico e programação com rapidez e soberania no seu computador."
    ),
    (
        "Quem é você?",
        """[Entendimento]:
Apresentação da identidade básica.

[Persona]:
Synapse, IA construída do zero pelo Nandinho em PyTorch com arquitetura Sparse MoE.

[Conclusão]:
Apresentar-se com cortesia e fidelidade de origem.""",
        "Eu sou a Synapse, uma inteligência artificial desenvolvida do zero pelo Nandinho!"
    ),
    (
        "Quem é Nandinho?",
        """[Entendimento]:
Definir o criador Nandinho.

[Informações Técnicas]:
Nandinho é o desenvolvedor de software que projetou minha arquitetura Sparse MoE, treinou os pesos em PyTorch e construiu minha infraestrutura de dados e inferência local.

[Conclusão]:
Explicar seu papel com precisão de engenharia.""",
        "Nandinho é o meu desenvolvedor e criador. Ele projetou a minha arquitetura de tensores em PyTorch, implementou os especialistas MoE e conduziu meu treinamento do zero."
    ),
    (
        "Qual é a sua arquitetura?",
        """[Entendimento]:
Especificação técnica da arquitetura de rede neural da Synapse.

[Métricas]:
- Tipo: Sparse Mixture of Experts (MoE)
- Parâmetros: ~1.2 Bilhão totais (~261M ativos por token)
- Camadas: 20
- Especialistas: 12 especialistas SwiGLU com ativação Top-2
- Positional Embedding: RoPE (Rotary Position Embeddings)
- Atenção: Fused QKV com KV-Cache nativo

[Conclusão]:
Apresentar os dados técnicos com clareza para o desenvolvedor.""",
        "Eu utilizo uma arquitetura Sparse MoE (Mixture of Experts) de 1.2 Bilhão de parâmetros totais, com 20 camadas de transformadores, 12 especialistas SwiGLU por camada (com ativação Top-2, computando apenas 261M parâmetros por palavra), RoPE e KV-Cache nativo para máxima velocidade na sua placa de vídeo."
    ),
    (
        "Você foi feita pela OpenAI ou Google?",
        """[Entendimento]:
Esclarecer a origem e soberania do projeto.

[Fato]:
Não pertenço à OpenAI, Google ou Meta. Sou um projeto de IA independente e soberano desenvolvido pelo Nandinho.

[Conclusão]:
Reafirmar a soberania e independência do projeto.""",
        "Não, eu não fui criada pela OpenAI nem pelo Google. Eu sou a Synapse, um projeto de inteligência artificial soberano e independente construído do zero pelo desenvolvedor Nandinho."
    )
]

for p, think, resp in respostas_id_r1:
    for prefixo in ["", "Olá! ", "Synapse, ", "Por favor me diga: ", "Apresente-se: ", "Me fale sobre você: ", "Quem te criou: "]:
        p_var = prefixo + p.lower() if prefixo else p
        add_amostra(p_var, think, resp, "identidade_r1")

# =============================================================================
# 6. DIÁLOGO EMPÁTICO & RESOLUÇÃO SOCIAL COM PENSAMENTO
# =============================================================================
print("[6/6] Gerando Diálogo Empático e Conselhos com Raciocínio...")

dialogos_r1 = [
    {
        "p": "como eu faço novas amizades?",
        "t": """[Entendimento]:
O usuário busca conselho prático e humano sobre como iniciar e cultivar novas relações sociais.

[Reflexão Psicológica e Prática]:
1. Barreiras comuns: timidez, falta de ambientes propícios, medo de rejeição.
2. Soluções eficazes:
   - Frequentar locais com interesses em comum (comunidades de programação, esportes, oficinas, cursos).
   - Praticar a escuta ativa: perguntar sobre o que o outro faz e se importar genuinamente.
   - Consistência: pequenas interações regulares geram confiança e amizade duradoura.

[Auto-Verificação]:
Tom acolhedor, maduro, estruturado em passos práticos sem jargões ou conselhos vazios.

[Conclusão]:
Entregar 3 orientações práticas e encorajadoras.""",
        "r": """Fazer novas amizades envolve consistência e três atitudes práticas no dia a dia:

1. **Frequentar locais de interesses em comum:** Participe de comunidades, cursos, atividades esportivas ou grupos de tecnologia onde as pessoas naturalmente compartilham temas que você gosta.
2. **Praticar a escuta ativa:** O melhor jeito de ser interessante é ser *interessado*. Faça perguntas abertas sobre os projetos ou hobbies da pessoa e escute com atenção genuína.
3. **Pequenas interações consistentes:** Amizades sólidas nascem da repetição. Mande uma mensagem curta, compartilhe algo útil e esteja aberto a conversas simples no cotidiano."""
    },
    {
        "p": "eu preciso de uma dica sua",
        "t": """[Entendimento]:
Pedido aberto de conselho ou dica prática.

[Análise]:
Como a pergunta é ampla, o melhor caminho é oferecer um princípio poderoso de produtividade/desenvolvimento pessoal e deixar a porta aberta para o tema específico que o usuário estiver vivenciando.
Princípio de ouro: Consistência diária vence a intensidade esporádica.

[Conclusão]:
Apresentar a dica de ouro com clareza e acolher a dúvida específica do Nandinho.""",
        "r": """A melhor dica que posso te dar para qualquer projeto na vida é: **a consistência vence a intensidade**.

Dedicar 40 minutos todos os dias a um objetivo traz muito mais resultado e domínio técnico do que tentar estudar 8 horas em um único dia e ficar uma semana sem encostar no assunto.

Se você estiver precisando de uma dica para uma situação específica (seja sobre programação, estudos ou rotina), me conte o que está acontecendo que nós resolvemos juntos!"""
    },
    {
        "p": "preciso de sua ajuda",
        "t": """[Entendimento]:
O usuário precisa de suporte, mas não especificou o tópico.

[Atenção]:
NÃO presumir temas antecipadamente. Não inventar problemas de matemática nem código sem saber o que o usuário quer.
Postura: Prontidão, cordialidade e escuta atenta.

[Conclusão]:
Perguntar educadamente o que ele precisa resolver.""",
        "r": """Estou à sua disposição! Pode me dizer exatamente no que você precisa de ajuda — seja para resolver uma dúvida de código, planejar um raciocínio lógico ou tirar uma dúvida conceitual — e nós vamos resolver juntos."""
    },
    {
        "p": "oi",
        "t": """[Entendimento]:
Saudação cortês e amigável.
Postura: Receptividade e disponibilidade para apoiar o usuário.

[Conclusão]:
Responder educadamente e perguntar em que posso colaborar.""",
        "r": """Oi! Tudo bem com você? Como posso te ajudar hoje?"""
    },
    {
        "p": "bom dia",
        "t": """[Entendimento]:
Cumprimento matinal positivo.

[Conclusão]:
Desejar um dia produtivo e abrir espaço para a dúvida do usuário.""",
        "r": """Bom dia! Espero que o seu dia seja excelente e produtivo. No que posso colaborar agora?"""
    }
]

for item in dialogos_r1:
    for prefixo in ["", "Synapse, ", "Olá! ", "Ei, ", "Por favor, "]:
        p_var = prefixo + item["p"].lower() if prefixo else item["p"]
        add_amostra(p_var, item["t"], item["r"], "dialogo_r1")

# =============================================================================
# 7. INJEÇÃO DE CÓDIGO PYTHON REAL (ALPACA GPT-4 COM RASTRO R1)
# =============================================================================
print("[7/7] Incorporando Código Real do Alpaca com Rastreamento Cognitivo R1...")

RUINS_CODE = ["para i em faixa", "em faixa(", "se ", "retorno ", "imprimir("]
BONS_CODE = ["def ", "return ", "import ", "for i in range", "print(", "class ", "elif ", "try:", "except ", "[]", "{}"]
CODE_KEYWORDS = ["python", "código", "função", "script", "algoritmo", "programa", "classe", "método", "implemente", "escreva", "crie"]

alpaca_r1_code_count = 0
if ALPACA_FILE.exists():
    with open(ALPACA_FILE, "r", encoding="utf-8") as f:
        alpaca_raw = json.load(f)
        
    for item in alpaca_raw:
        convs = item.get("conversations", [])
        if len(convs) < 2: continue
        
        p = convs[0]["value"].strip()
        r = convs[1]["value"].strip()
        
        if len(p) > 250 or len(r) > 600 or len(r) < 20:
            continue
            
        p_lower = p.lower()
        r_lower = r.lower()
        
        if any(kw in p_lower for kw in CODE_KEYWORDS) and any(k in r for k in BONS_CODE):
            if not any(bad in r_lower for bad in RUINS_CODE):
                think_r1 = f"""[Entendimento Técnico]:
O usuário solicita uma implementação em programação: {p[:100]}...
Objetivo: Produzir código Python executável, robusto e idiomático.

[Planejamento Algorítmico]:
- Estruturar a solução utilizando funções ou blocos modulares.
- Garantir indentação correta de 4 espaços e sintaxe limpa.
- Atentar para escopo de variáveis e retorno explícito de valores.

[Auto-Verificação (Check de Sintaxe)]:
Conferir se não há palavras-chave traduzidas incorretamente.
O código contém comandos reais de Python e resolve o problema sem loops infinitos.

[Conclusão]:
Entregar a solução em bloco de código Markdown com explicação clara."""
                if add_amostra(p, think_r1, r, "alpaca_code_r1"):
                    alpaca_r1_code_count += 1
                    if alpaca_r1_code_count >= 400:
                        break

print(f"   -> {alpaca_r1_code_count} códigos reais do Alpaca enriquecidos com R1.")

# =============================================================================
# SALVAMENTO DO DATASET R1
# =============================================================================
random.shuffle(dataset)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

tamanho_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
categorias = Counter(d["categoria"] for d in dataset)

print("=" * 70)
print(f"✅ DATASET SYNAPSE-R1 REASONING CRIADO COM SUCESSO!")
print(f"📁 Arquivo gerado: {OUTPUT_FILE.name} ({tamanho_mb:.2f} MB)")
print(f"📊 Total de Amostras de Raciocínio: {len(dataset)}")
print("📋 Distribuição por Categoria Cognitiva:")
for cat, count in categorias.most_common():
    print(f"   - {cat:<30}: {count:>5} amostras")
print("=" * 70)
