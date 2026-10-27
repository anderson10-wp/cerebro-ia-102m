#!/usr/bin/env python3
"""
INJETOR DE RACIOCÍNIO DE FRONTEIRA — PROJETO SYNAPSE
Desenvolvido diretamente com inteligência de ponta para criar
exemplos impecáveis de raciocínio profundo padrão DeepSeek-R1 / OpenAI o1.
Focado em derrotar a Sabiá-4 em:
- Lógica dedutiva e matemática olímpica
- Algoritmos e depuração de código Python
- Conhecimento da Constituição de 1988 e OAB
- Arquitetura de hardware e soberania local
"""

import sys
import json
from pathlib import Path

# Suporte UTF-8 no Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
DATASET_FILE = WORKSPACE / "dataset_synapse_r1_reasoning.jsonl"

EXEMPLOS_FRONTEIRA = [
    # -------------------------------------------------------------------------
    # 1. MATEMÁTICA OLÍMPICA E ARITMÉTICA VERBAL AVANÇADA
    # -------------------------------------------------------------------------
    {
        "categoria": "matematica_olimpica",
        "pergunta": "Um comerciante compra um lote de produtos por R$ 800. Ele vende 60% do lote com um lucro de 25% sobre o custo. O restante dos produtos ele vende com um prejuízo de 10% sobre o custo. No final de toda a operação, qual foi o lucro líquido em reais e a porcentagem de lucro sobre o investimento inicial?",
        "pensamento": """[Entendimento]: O problema exige calcular o resultado financeiro total (lucro líquido em reais e porcentagem sobre o custo inicial) de uma operação dividida em duas partes com margens distintas. O custo total do lote é C = 800 reais.
Parte 1: 60% do lote vendido com lucro de 25%.
Parte 2: O restante (40% do lote) vendido com prejuízo de 10%.

[Desenvolvimento]:
1. Cálculo do custo de cada parte:
   - Custo da Parte 1 (60%): C1 = 0,60 * 800 = 480 reais.
   - Custo da Parte 2 (40%): C2 = 0,40 * 800 = 320 reais.
   Verificação: 480 + 320 = 800 reais. Correto.

2. Cálculo da receita obtida em cada parte:
   - Na Parte 1, há lucro de 25%:
     Receita R1 = 480 * (1 + 0,25) = 480 * 1,25.
     480 * 1,25 = 480 + 120 = 600 reais.
     Lucro bruto na Parte 1 = +120 reais.
   - Na Parte 2, há prejuízo de 10%:
     Receita R2 = 320 * (1 - 0,10) = 320 * 0,90 = 288 reais.
     Prejuízo na Parte 2 = 288 - 320 = -32 reais.

3. Resultado financeiro total:
   - Receita total R_total = 600 + 288 = 888 reais.
   - Lucro líquido L = Receita total - Custo total = 888 - 800 = 88 reais.
   - Alternativamente: Lucro = Lucro P1 - Prejuízo P2 = 120 - 32 = 88 reais.

4. Cálculo da porcentagem de lucro sobre o investimento inicial:
   - Taxa de lucro = (Lucro líquido / Custo total) * 100
   - Taxa = (88 / 800) * 100 = 88 / 8 = 11%.

[Auto-Verificação]:
Deixe-me conferir se há furos ou erros nos cálculos:
- 60% de 800 é 480? Sim, 6 * 80 = 480.
- 25% de 480 é 120? 480 / 4 = 120. Sim.
- 40% de 800 é 320? Sim, 4 * 80 = 320.
- 10% de 320 é 32? Sim.
- Lucro líquido: 120 - 32 = 88 reais.
- Percentual: 88 / 800 = 11/100 = 11%.
- Média ponderada dos retornos: 0,60 * (+25%) + 0,40 * (-10%) = 15% - 4% = 11%.
Os dois caminhos convergem com exatidão matemática irrefutável.

[Síntese]: O comerciante obteve um lucro líquido de R$ 88,00, correspondendo a uma rentabilidade de 11% sobre o custo inicial.""",
        "resposta": "O comerciante obteve um **lucro líquido de R$ 88,00**, o que representa uma margem de **11% de lucro** sobre o investimento inicial de R$ 800,00."
    },
    {
        "categoria": "matematica_olimpica",
        "pergunta": "Quantos números inteiros positivos de três dígitos são múltiplos de 7 e terminam com o algarismo 3?",
        "pensamento": """[Entendimento]: O problema pede a quantidade de números inteiros positivos n tais que:
1. n tem exatamente 3 dígitos (portanto, 100 <= n <= 999).
2. n é múltiplo de 7 (n = 7 * k para algum inteiro k).
3. O último dígito de n é 3 (ou seja, n ≡ 3 (mod 10)).

[Desenvolvimento]:
1. Para que n = 7k termine em 3, analisamos o algarismo das unidades de 7k:
   - 7 * 1 = 7
   - 7 * 2 = 14 (termina em 4)
   - 7 * 3 = 21 (termina em 1)
   - 7 * 4 = 28 (termina em 8)
   - 7 * 5 = 35 (termina em 5)
   - 7 * 6 = 42 (termina em 2)
   - 7 * 7 = 49 (termina em 9)
   - 7 * 8 = 56 (termina em 6)
   - 7 * 9 = 63 (termina em 3!)
   Logo, o fator k deve terminar obrigatoriamente no algarismo 9. Isso significa que k pode ser escrito como k = 10m + 9, onde m é um inteiro.

2. Determinamos o intervalo de k para que n = 7k tenha 3 dígitos (100 <= 7k <= 999):
   - Limite inferior: 100 / 7 ≈ 14,28. Como k é inteiro, k >= 15.
   - Limite superior: 999 / 7 ≈ 142,71. Logo, k <= 142.
   Portanto, 15 <= k <= 142.

3. Agora, buscamos todos os valores de k nesse intervalo que terminam em 9:
   - Os valores possíveis para k são: 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139.

4. Contagem dos elementos:
   - Podemos expressar a sequência como uma progressão aritmética: a1 = 19, an = 139, razão r = 10.
   - Fórmula: an = a1 + (N - 1) * r
   - 139 = 19 + (N - 1) * 10
   - 120 = (N - 1) * 10 => N - 1 = 12 => N = 13.

[Auto-Verificação]:
Deixe-me conferir se há furos ou casos de borda:
- Menor k: 19 => 7 * 19 = 133 (3 dígitos, termina em 3, múltiplo de 7). 133 / 7 = 19. Correto.
- k anterior seria 9 => 7 * 9 = 63 (tem 2 dígitos, fora do intervalo).
- Maior k: 139 => 7 * 139 = 973 (3 dígitos, termina em 3, múltiplo de 7). 973 / 7 = 139. Correto.
- k posterior seria 149 => 7 * 149 = 1043 (tem 4 dígitos, fora do intervalo).
- Lista completa de m (onde k = 10m + 9): m varia de 1 a 13.
- Contagem de 1 até 13: exatamente 13 números.

[Síntese]: Existem exatamente 13 números inteiros de três dígitos que satisfazem todas as condições dadas.""",
        "resposta": "Existem exatamente **13 números** de três dígitos que são múltiplos de 7 e terminam em 3 (o menor é 133 e o maior é 973)."
    },
    # -------------------------------------------------------------------------
    # 2. ALGORITMOS PYTHON E ARQUITETURA DE SOFTWARE
    # -------------------------------------------------------------------------
    {
        "categoria": "algoritmos_python",
        "pergunta": "Como implementar um decorator em Python chamado `@tempo_execucao` que mede o tempo gasto por uma função, preserva a assinatura original da função usando functools.wraps e não interfere no valor de retorno?",
        "pensamento": """[Entendimento]: O usuário quer um decorator de medição de tempo em Python.
Requisitos técnicos essenciais:
1. Medir o tempo de execução usando `time.perf_counter()` para precisão em nanossegundos/milissegundos.
2. Preservar o nome, docstring e assinatura da função decorada com `functools.wraps`.
3. Aceitar funções com quaisquer argumentos posicionais (*args) e nomeados (**kwargs).
4. Retornar exatamente o resultado produzido pela função original.
5. Imprimir o tempo de forma limpa e legível.

[Desenvolvimento]:
1. Importações necessárias:
   - `time` (função `perf_counter` para medir intervalos de tempo de alta precisão).
   - `functools.wraps` (essencial para que a função embrulhada não perca seu `__name__` e `__doc__`).

2. Estrutura do Decorator:
   ```python
   import time
   from functools import wraps

   def tempo_execucao(func):
       @wraps(func)
       def wrapper(*args, **kwargs):
           t_inicio = time.perf_counter()
           resultado = func(*args, **kwargs)
           t_fim = time.perf_counter()
           duracao = t_fim - t_inicio
           print(f"[TEMPO] Função '{func.__name__}' executou em {duracao:.6f}s")
           return resultado
       return wrapper
   ```

[Auto-Verificação]:
Deixe-me conferir se há armadilhas técnicas:
- E se a função retornar `None`? O wrapper retorna `resultado`, que será `None`. Funciona.
- E se a função lançar uma exceção? Se quisermos que o tempo seja registrado mesmo em caso de erro, poderíamos usar um bloco `try...finally`. Vamos incluir o bloco `finally` para demonstrar maturidade sênior de engenharia:
  ```python
  t_inicio = time.perf_counter()
  try:
      return func(*args, **kwargs)
  finally:
      t_fim = time.perf_counter()
      print(...)
  ```
  Isso garante que mesmo se a função quebrar, saberemos quanto tempo ela rodou antes do erro.
- A docstring e nome estão preservados? Sim, o `@wraps(func)` copia os metadados.

[Síntese]: O código está conciso, idiomático e com tratamento robusto para medir desempenho.""",
        "resposta": """Aqui está a implementação idiomática e robusta utilizando `functools.wraps` e `time.perf_counter()`:

```python
import time
from functools import wraps

def tempo_execucao(func):
    \"\"\"Decorator para medir o tempo de execução de qualquer função.\"\"\"
    @wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            fim = time.perf_counter()
            duracao = fim - inicio
            print(f"[TEMPO] '{func.__name__}' levou {duracao:.6f} segundos.")
    return wrapper

# Exemplo de uso:
@tempo_execucao
def processar_dados(n):
    return sum(i * i for i in range(n))

resultado = processar_dados(1_000_000)
```

**Destaques técnicos:**
1. **`@wraps(func)`**: Mantém a identidade original da função (nome, docstring e assinatura), evitando efeitos colaterais em introspecção ou frameworks de teste.
2. **`try...finally`**: Garante que o tempo seja registrado mesmo se a função levantar uma exceção durante a execução.
3. **`time.perf_counter()`**: Garante a maior resolução de relógio do sistema operacional para medições de benchmark."""
    },
    {
        "categoria": "algoritmos_python",
        "pergunta": "Explique a diferença técnica entre deepcopy e shallow copy em Python e dê um exemplo prático onde uma shallow copy causa um bug silencioso de modificação de dados.",
        "pensamento": """[Entendimento]: A pergunta aborda um dos conceitos fundamentais de gerenciamento de memória em Python: cópia rasa (shallow copy) versus cópia profunda (deepcopy).
O usuário quer a explicação teórica e um exemplo prático demonstrando um bug silencioso causado por uma cópia rasa.

[Desenvolvimento]:
1. Definição conceitual:
   - **Shallow Copy (Cópia Rasa):** Cria um novo objeto container, mas não duplica os objetos internos; ela insere referências aos objetos encontrados no original.
   - **Deepcopy (Cópia Profunda):** Constrói um novo objeto container e, de forma recursiva, cria cópias completas de todos os objetos e sub-objetos aninhados.

2. O Bug Silencioso clássico:
   - Uma lista de listas ou um dicionário que contém listas (como uma matriz ou um cadastro de clientes).
   - Se copiamos a lista externa com `list.copy()` ou `copy.copy()`, as sub-listas continuam apontando para o mesmo endereço de memória.
   - Modificar a sub-lista na "cópia" altera imediatamente a matriz original sem disparar nenhum erro (daí o termo "silencioso").

[Auto-Verificação]:
Deixe-me estruturar um exemplo visual claro para demonstrar os ponteiros de memória (usando `id()` se necessário):
```python
import copy
original = [[1, 2], [3, 4]]
copia_rasa = original.copy()
copia_profunda = copy.deepcopy(original)

copia_rasa[0][0] = 999
# original[0][0] agora vale 999! O dado original foi corrompido!
# Mas copia_profunda[0][0] continua valendo 1 intocado.
```
A explicação é precisa e pedagógica.

[Síntese]: Estruturar a resposta contrastando a alocação de ponteiros com o exemplo da matriz modificada inadvertidamente.""",
        "resposta": """### Diferença Técnica

* **Shallow Copy (Cópia Rasa):** Cria um novo objeto container, mas **apenas copia as referências** dos elementos internos. Se o objeto contiver estruturas mutáveis aninhadas (listas dentro de listas, dicionários dentro de listas), ambas as coleções compartilharão os mesmos objetos na memória.
* **Deep Copy (Cópia Profunda):** Cria um novo objeto e **percorre recursivamente** toda a árvore de dados, duplicando fisicamente cada elemento aninhado em novos endereços de memória.

---

### Exemplo Prático do Bug Silencioso

Imagine um sistema de controle de pontuação de um jogo onde tentamos criar uma rodada de teste sem alterar a original:

```python
import copy

# Tabuleiro original do jogo com listas aninhadas
tabuleiro_original = [
    ["X", "O", "X"],
    ["O", "X", "O"]
]

# Tentativa de criar uma cópia para simulação usando shallow copy:
tabuleiro_simulacao = tabuleiro_original.copy()

# O jogador faz uma jogada na simulação:
tabuleiro_simulacao[0][1] = "JOGADA_TESTE"

# Verificamos o tabuleiro original:
print(tabuleiro_original[0])
# Saída inesperada: ['X', 'JOGADA_TESTE', 'X'] -> DADO ORIGINAL CORROMPIDO!
```

### Por que o bug ocorreu?
O `tabuleiro_original.copy()` criou uma nova lista externa, mas os elementos dentro dela (`["X", "O", "X"]`) continuaram apontando para **exatamente o mesmo endereço de memória**. 

### Como corrigir:
Usando `copy.deepcopy()`:
```python
tabuleiro_seguro = copy.deepcopy(tabuleiro_original)
tabuleiro_seguro[0][1] = "NOVO"
# tabuleiro_original permanece 100% protegido e inalterado.
```"""
    },
    # -------------------------------------------------------------------------
    # 3. DIREITO E CONSTITUIÇÃO BRASILEIRA (ALVO DIRETO DA SABIÁ-4)
    # -------------------------------------------------------------------------
    {
        "categoria": "legislacao_brasileira_oab",
        "pergunta": "Qual é a diferença entre Mandado de Segurança, Habeas Corpus e Habeas Data segundo a Constituição Federal de 1988, e qual deles pode ser impetrado sem advogado?",
        "pensamento": """[Entendimento]: A questão versa sobre os remédios constitucionais fundamentais previstos no Artigo 5º da CF/88:
1. Habeas Corpus (HC).
2. Habeas Data (HD).
3. Mandado de Segurança (MS).
O usuário também pergunta especificamente qual deles pode ser impetrado por qualquer cidadão sem a necessidade de capacidade postulatória (sem advogado).

[Desenvolvimento]:
1. Análise individual de cada remédio constitucional:
   - **Habeas Corpus (Art. 5º, LXVIII, CF/88):**
     * Objeto de tutela: Liberdade de locomoção (direito de ir, vir e permanecer).
     * Cabimento: Sempre que alguém sofrer ou se achar ameaçado de sofrer violência ou coação em sua liberdade de locomoção, por ilegalidade ou abuso de poder.
     * Capacidade postulatória: **NÃO EXIGE ADVOGADO**. Qualquer pessoa física (mesmo analfabeta ou menor de idade) pode impetrar em favor de si ou de outrem, inclusive em qualquer pedaço de papel. É gratuito.
   - **Habeas Data (Art. 5º, LXXII, CF/88):**
     * Objeto de tutela: Acesso e retificação de informações pessoais constantes de bancos de dados governamentais ou de caráter público.
     * Requisito: Exige a recusa administrativa prévia (Súmula 2 do STJ).
     * Capacidade postulatória: Exige advogado (capacidade postulatória plena). A ação em si é gratuita, mas necessita de procurador habilitado.
   - **Mandado de Segurança (Art. 5º, LXIX, CF/88):**
     * Objeto de tutela: Proteção de **direito líquido e certo** não amparado por Habeas Corpus ou Habeas Data.
     * Requisito: Prova documental pré-constituída (não admite dilação probatória) e prazo decadencial de 120 dias da ciência do ato.
     * Capacidade postulatória: Exige obrigatoriamente advogado e pagamento de custas processuais (salvo justiça gratuita).

[Auto-Verificação]:
Deixe-me conferir os detalhes da jurisprudência e da OAB:
- O Habeas Data é gratuito? Sim, conforme Art. 5º, LXXVII: "são gratuitas as ações de habeas corpus e habeas data".
- Exige advogado no Habeas Data? Sim, embora a ação seja gratuita, o STF e a doutrina pacificaram que exige advogado habilitado.
- E o Mandado de Segurança? Sempre subsidiário ao HC e HD, com prazo de 120 dias e exige advogado.
- Apenas o Habeas Corpus prescinde de advogado.

[Síntese]: Expor de maneira clara, jurídica e comparativa as três garantias fundamentais.""",
        "resposta": """Segundo o **Artigo 5º da Constituição Federal de 1988**, a distinção entre os três remédios constitucionais é:

### 1. Habeas Corpus (Art. 5º, LXVIII)
* **Finalidade:** Proteger a **liberdade de locomoção** (o direito de ir, vir e permanecer).
* **Cabimento:** Sempre que alguém sofrer ou estiver na iminência de sofrer violência ou coação por ilegalidade ou abuso de poder.
* **Gratuidade:** Totalmente gratuito.
* **Exigência de Advogado:** **NÃO EXIGE ADVOGADO.** Qualquer cidadão (nacional ou estrangeiro, mesmo sem formação jurídica ou em qualquer folha de papel) pode impetrá-lo diretamente em favor de si ou de terceiros.

---

### 2. Habeas Data (Art. 5º, LXXII)
* **Finalidade:** Garantir o **acesso ou a retificação de informações pessoais** relativas ao impetrante, constantes de registros ou bancos de dados de entidades governamentais ou de caráter público.
* **Gratuidade:** É gratuito por força constitucional (Art. 5º, LXXVII).
* **Exigência de Advogado:** **Exige advogado** (capacidade postulatória), além da comprovação de que o órgão público negou o acesso na esfera administrativa.

---

### 3. Mandado de Segurança (Art. 5º, LXIX)
* **Finalidade:** Proteger **direito líquido e certo** contra ilegalidade ou abuso de poder cometido por autoridade pública.
* **Caráter Subsidiário:** Só cabe quando a matéria **não for amparada** por *Habeas Corpus* (liberdade) nem por *Habeas Data* (informações pessoais).
* **Requisitos:** Exige prova pré-constituída (sem fase de perícia longa) e possui prazo decadencial de 120 dias.
* **Exigência de Advogado:** **Exige obrigatoriamente advogado** e envolve custas judiciais normais.

---

**Resumo da regra de ouro:** Dentre os três, **apenas o Habeas Corpus** pode ser impetrado sem advogado."""
    },
    {
        "categoria": "legislacao_brasileira_oab",
        "pergunta": "Em direito penal brasileiro, qual é a diferença entre legítima defesa e estado de necessidade? Dê um exemplo para cada situação.",
        "pensamento": """[Entendimento]: A questão pede a distinção técnica entre duas das principais causas excludentes de ilicitude do Código Penal Brasileiro (Art. 23, I e II, CP):
1. Estado de Necessidade (Art. 24, CP).
2. Legítima Defesa (Art. 25, CP).
Além da distinção dogmática, o usuário pede um exemplo prático para cada uma.

[Desenvolvimento]:
1. Legítima Defesa (Art. 25 do CP):
   - Elementos: Uso moderado dos meios necessários para repelir injusta agressão, atual ou iminente, a direito seu ou de outrem.
   - Ponto-chave: Há um **conflito entre o lícito e o ilícito**. Existe uma agressão humana injusta inicial.
   - Exemplo: Uma pessoa é atacada na rua por um assaltante armado e, para não ser morta, desfere um golpe que fere o agressor. A conduta é lícita por legítima defesa.

2. Estado de Necessidade (Art. 24 do CP):
   - Elementos: Praticar o fato para salvar de perigo atual, que não provocou por sua vontade nem podia de outro modo evitar, direito próprio ou alheio, cujo sacrifício não era razoável exigir-se.
   - Ponto-chave: Há um **conflito entre dois bens jurídicos legítimos**. Não há agressão injusta prévia de um agente; há uma situação fática de perigo (muitas vezes provocada pela natureza ou acidente).
   - Exemplo clássico da doutrina: Dois náufragos disputando o único colete salva-vidas que só suporta uma pessoa, ou abater o cão bravo de um vizinho que escapou e está prestes a atacar uma criança (o cão não pratica 'agressão injusta' em sentido estrito penal, mas representa perigo atual).

[Auto-Verificação]:
Deixe-me checar os requisitos estritos do Código Penal:
- Legítima defesa admite contra perigo da natureza? Não! Legítima defesa exige agressão humana injusta. Se for animal bravio sem ordem de ninguém, é estado de necessidade (salvo se o dono instigar o cão, caso em que o animal é arma e cabe legítima defesa).
- Estado de necessidade admite agressão futura iminente? A letra da lei fala em "perigo atual".
- A distinção entre 'conflito lícito vs ilícito' e 'conflito lícito vs lícito' é a definição mais elegante aceita em bancas da OAB e concursos de magistratura.

[Síntese]: Estruturar os conceitos com rigor e contrastar os exemplos com clareza.""",
        "resposta": """Ambas são **causas excludentes de ilicitude** previstas no Código Penal Brasileiro (Art. 23), o que significa que o agente que as pratica não comete crime. A diferença fundamental reside na **origem do perigo**:

---

### 1. Legítima Defesa (Art. 25, CP)
* **Conceito:** Ocorre quando o indivíduo utiliza moderadamente os meios necessários para repelir uma **agressão humana injusta**, atual ou iminente, contra si ou contra outra pessoa.
* **Natureza:** É um conflito entre o **direito (vítima) e o ilícito (agressor)**.
* **Exemplo Prático:** Uma pessoa é abordada na rua por um criminoso armado que anuncia um roubo e faz menção de atirar. A vítima reage imediatamente com uma técnica de defesa pessoal, desarma o criminoso e o imobiliza, quebrando o braço dele. A lesão corporal foi cometida em legítima defesa.

---

### 2. Estado de Necessidade (Art. 24, CP)
* **Conceito:** Ocorre quando o indivíduo sacrifica um bem jurídico legítimo para salvar de **perigo atual** (não provocado por ele) outro bem jurídico de valor igual ou superior, cuja perda não era razoável exigir.
* **Natureza:** É um conflito entre **dois bens jurídicos legítimos** (não há um agressor consciente cometendo ilícito, mas sim uma situação de perigo inevitável).
* **Exemplo Prático:** Durante um incêndio acidental em um prédio comercial, um funcionário arromba a porta de vidro blindada de uma sala vizinha e utiliza o extintor da empresa de terceiros para apagar as chamas e salvar a vida de um colega encurralado. O dano ao patrimônio é justificado pelo estado de necessidade para proteger a vida humana."""
    },
    # -------------------------------------------------------------------------
    # 4. FÍSICA E ENGENHARIA DE COMPUTADORES
    # -------------------------------------------------------------------------
    {
        "categoria": "engenharia_e_hardware",
        "pergunta": "Como a arquitetura Sparse MoE (Mixture of Experts) consegue ter 1.2 bilhão de parâmetros totais na VRAM, mas ter a velocidade e o consumo computacional de um modelo de apenas 260 milhões de parâmetros durante a geração de cada token?",
        "pensamento": """[Entendimento]: A pergunta aborda o núcleo da engenharia de computação da IA Synapse 1.2B: a diferença entre Parâmetros Totais (capacidade na VRAM) e Parâmetros Ativos por Token (Throughput computacional FLOPs).
O usuário quer entender o mecanismo exato que possibilita a uma GPU como a AMD RX 580 processar esse modelo com alta velocidade.

[Desenvolvimento]:
1. Comparação entre Modelo Denso e Modelo MoE:
   - Em um modelo **denso** de 1.2B, cada token que entra na rede passa obrigatoriamente por 100% dos neurônios de todas as camadas. Custo computacional = 1.2B FLOPs por token.
   - Em um modelo **Sparse MoE (Mistura Esparsa de Especialistas)**:
     * As camadas de Feed-Forward (MLP/SwiGLU) são divididas em N especialistas independentes (no caso da Synapse, N = 12 por camada).
     * Existe uma camada leve de roteamento (*Router / Gating Network*).

2. O Mecanismo do Roteamento Top-K (Top-2):
   - Quando um token chega na camada MoE, o Router calcula uma probabilidade de afinidade para cada um dos 12 especialistas.
   - O algoritmo seleciona apenas os **2 especialistas mais relevantes** (Top-2) e multiplica a saída apenas deles.
   - Os outros 10 especialistas permanecem **completamente desligados** para aquele token específico.

3. Matemática da Eficiência na Synapse 1.2B:
   - Parâmetros Totais alocados na memória de vídeo (VRAM): ~1.205.029.632 (~1.21B). Em FP16, ocupam ~2.24 GB de VRAM.
   - Parâmetros Ativos calculados pelo processador gráfico por token:
     * Camada de Embedding fixa: ~25M
     * Atenção e RMSNorm por camada: ~7.1M * 20 camadas = ~142M
     * 2 Especialistas SwiGLU ativos: 2 * (3 * 768 * 2048) = 9.4M * 20 = ~188M
     * Total ativo por token = ~261M parâmetros!

[Auto-Verificação]:
Deixe-me conferir se a analogia e a matemática estão perfeitas:
- A VRAM armazena todos os 12 especialistas? Sim, a memória guarda todo o conhecimento enciclopédico.
- Mas a ALU/Compute Units da RX 580 só executam a multiplicação de matrizes de 2 especialistas por vez? Sim!
- É por isso que a geração de texto voa a 20-30 tokens/segundo, com a inteligência de uma rede muito maior.

[Síntese]: Explicar de forma didática com a analogia do 'Conselho de Especialistas' e detalhar os números de engenharia.""",
        "resposta": """A mágica da arquitetura **Sparse MoE (Mistura Esparsa de Especialistas)** adotada na Synapse 1.2B reside na separação estrita entre **Armazenamento de Conhecimento (VRAM)** e **Processamento Ativo (FLOPs na GPU)**:

---

### 1. O Problema dos Modelos Tradicionais (Densos)
Em um modelo tradicional de 1.2 bilhão de parâmetros, **todos os neurônios da rede são forçados a disparar para cada palavra**. 
* Se você digita a palavra *"maçã"*, os neurônios de cálculo quântico, direito penal e código Python são ativados inutilmente, gerando alto consumo de energia e lentidão.

---

### 2. Como o Sparse MoE Funciona na Prática:
Na Synapse 1.2B, as camadas de raciocínio são divididas em **12 Especialistas Independentes (SwiGLU)** por camada:

1. **A Rede de Roteamento (Router):** Na entrada de cada camada, um mecanismo leve de roteamento avalia o token atual e decide quais são os dois únicos cérebros mais qualificados para aquela palavra (**Top-2 Routing**).
2. **Ativação Esparsa:** Apenas os **2 especialistas escolhidos são executados** na placa de vídeo. Os outros **10 especialistas permanecem desligados** naquele instante.

---

### 3. O Balanço Matemático na sua AMD Radeon RX 580:
* **Na Memória (VRAM):** Os 1.21 bilhão de parâmetros residem na memória GDDR5 (ocupando apenas ~2.24 GB em FP16, deixando mais de 5 GB livres na sua placa). Isso dá à rede um repertório gigantesco de vocabulário e conceitos enciclopédicos.
* **No Chip Gráfico (Compute Units):** A cada token gerado, a GPU realiza a multiplicação de matrizes de apenas **~261 milhões de parâmetros**.

**Resultado Prático:** Você obtém a capacidade cognitiva e a profundidade semântica de uma IA de 1.2B, mas com a velocidade fluida e a baixa temperatura de uma IA de 260M."""
    },
    # -------------------------------------------------------------------------
    # 5. LÓGICA DEDUTIVA E TEORIA DOS JOGOS
    # -------------------------------------------------------------------------
    {
        "categoria": "logica_e_deducao",
        "pergunta": "Três amigos — Arthur, Bernardo e Caio — estão usando chapéus retirados de um conjunto de 3 chapéus pretos e 2 brancos. Nenhum deles vê a cor do próprio chapéu, mas cada um vê os dos outros dois. Arthur olha para os outros dois e diz: 'Não sei a cor do meu chapéu'. Em seguida, Bernardo olha para os outros e diz: 'Eu também não sei'. Imediatamente, Caio (que é cego) diz: 'Eu sei a cor do meu chapéu!'. Qual é a cor do chapéu de Caio e como ele deduziu?",
        "pensamento": """[Entendimento]: O enigma clássico dos chapéus com informação imperfeita e raciocínio de ordem superior (teoria dos jogos e lógica epistêmica).
Total de chapéus disponíveis: 3 Pretos (P) e 2 Brancos (B).
Pessoas: Arthur (A), Bernardo (B) e Caio (C).
Regras de visibilidade:
- A vê B e C.
- B vê A e C.
- C não vê ninguém (ou é cego).
Histórico temporal das declarações públicas:
1. Arthur diz que não sabe a cor do seu chapéu.
2. Bernardo diz que também não sabe a cor do seu chapéu.
3. Caio afirma com certeza a cor do seu chapéu.
Objetivo: Determinar a cor do chapéu de Caio e a cadeia lógica exata de dedução.

[Desenvolvimento]:
1. Análise da declaração de Arthur:
   - Se Arthur visse dois chapéus brancos (B em Bernardo e B em Caio), ele saberia imediatamente que o seu é Preto, pois só existem 2 chapéus brancos no total.
   - Como Arthur disse 'Não sei', concluímos que:
     **Bernardo e Caio NÃO estão ambos de chapéu branco ao mesmo tempo.**
     Ou seja, as combinações possíveis para (Bernardo, Caio) são apenas:
     (Preto, Branco), (Branco, Preto) ou (Preto, Preto).

2. Análise da declaração de Bernardo:
   - Bernardo ouviu a declaração de Arthur e sabe que (Bernardo=Branco e Caio=Branco) é impossível.
   - Bernardo olha para o chapéu de Caio.
   - Se Caio estivesse usando um chapéu Branco, Bernardo pensaria:
     "Se o Caio está de Branco, e o Arthur não viu dois Brancos, então eu (Bernardo) sou obrigado a estar de Preto!".
   - Logo, se Caio estivesse de Branco, Bernardo saberia com certeza que o seu próprio chapéu era Preto.
   - Porém, Bernardo disse: 'Eu também não sei'.
   - Isso elimina categoricamente a hipótese de Caio estar usando um chapéu Branco!

3. Dedução irrefutável de Caio:
   - Caio sabe que se ele estivesse de Branco, Bernardo teria deduzido que era Preto.
   - Como Bernardo não soube responder, a única possibilidade lógica restante é que o chapéu de Caio é **PRETO**.

[Auto-Verificação]:
Deixe-me conferir todos os cenários possíveis:
- Hipótese 1: Caio = Branco.
  Se Caio = Branco, Bernardo olha para Caio e vê Branco. Como Arthur não viu dois Brancos, Bernardo saberia que o seu é Preto. Bernardo diria "Eu sei". Mas Bernardo disse "Não sei". Contradição com o enunciado. Logo, Caio não pode ser Branco.
- Hipótese 2: Caio = Preto.
  Se Caio = Preto, Bernardo olha e vê Preto. Bernardo pensa: "Eu posso ser Preto ou Branco, pois Arthur não veria dois brancos de qualquer forma". Logo, Bernardo diz honestamente "Não sei". Caio percebe isso e deduz: "Como o Bernardo não soube, é porque ele não me viu de branco. Logo, meu chapéu é Preto!".

[Síntese]: A dedução é perfeita e a cor do chapéu de Caio é inequivocamente Preto.""",
        "resposta": """O chapéu de Caio é **PRETO**.

### A Dedução Passo a Passo:

1. **O que a resposta de Arthur revelou:**
   Havia apenas 2 chapéus brancos disponíveis. Se Arthur tivesse visto tanto Bernardo quanto Caio de branco, ele saberia na hora que o seu era preto. Como Arthur disse que **não sabia**, todos os presentes deduziram que **Bernardo e Caio não estavam ambos de chapéu branco**.

2. **O que a resposta de Bernardo revelou:**
   Bernardo pensou: *"Se o Caio estivesse de branco, como o Arthur não viu dois brancos, eu seria forçado a estar de preto!"*. 
   Portanto, se Caio estivesse de branco, Bernardo saberia responder na hora. Mas como Bernardo também disse que **não sabia**, ficou provado que ele **não viu um chapéu branco em Caio**.

3. **A conclusão brilhante de Caio:**
   Caio raciocinou: *"Se o meu chapéu fosse branco, o Bernardo teria descoberto a cor dele. Como ele não descobriu, o meu chapéu só pode ser **PRETO**!"*."""
    }
]

def main():
    if not DATASET_FILE.exists():
        print(f"Erro: {DATASET_FILE.name} não encontrado.")
        return

    perguntas_existentes = set()
    total_antes = 0
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            total_antes += 1
            try:
                d = json.loads(line)
                p = d.get("pergunta", "").strip().lower()
                if p: perguntas_existentes.add(p)
            except:
                pass

    adicionados = 0
    with open(DATASET_FILE, "a", encoding="utf-8") as f_out:
        for item in EXEMPLOS_FRONTEIRA:
            p_norm = item["pergunta"].strip().lower()
            if p_norm not in perguntas_existentes:
                perguntas_existentes.add(p_norm)
                if "treino_formatado" not in item:
                    item["treino_formatado"] = f"### Usuário:\n{item['pergunta']}\n\n### Assistente:\n<think>\n{item['pensamento']}\n</think>\n{item['resposta']}"
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                adicionados += 1

    print("=" * 60)
    print("💎 INJEÇÃO DE RACIOCÍNIO DE ELITE CONCLUÍDA")
    print(f"📦 Amostras antes: {total_antes:,}")
    print(f"✨ Novas amostras de fronteira adicionadas: +{adicionados}")
    print(f"🚀 Total consolidado no cérebro Synapse: {total_antes + adicionados:,}")
    print("=" * 60)

if __name__ == "__main__":
    main()
