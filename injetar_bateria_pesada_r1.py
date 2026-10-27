#!/usr/bin/env python3
"""
BATERIA PESADA DE RACIOCÍNIO R1 — TRANSCIAÇÃO NATIVA DE FRONTEIRA
Projetado para dar à Synapse 1.2B capacidade de superar a Sabiá-4 em:
- Matemática Avançada & Probabilidade
- Engenharia de Software Python & Baixo Nível
- Direito Constitucional e Penal Brasileiro (OAB / Magistratura)
- Física Teórica e Arquitetura de Processadores
"""

import sys
import json
from pathlib import Path

# Suporte UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
DATASET_FILE = WORKSPACE / "dataset_synapse_r1_reasoning.jsonl"

NOVOS_EXEMPLOS = [
    # -------------------------------------------------------------------------
    # MATEMÁTICA E PROBABILIDADE
    # -------------------------------------------------------------------------
    {
        "categoria": "probabilidade_e_estatistica",
        "pergunta": "No clássico Paradoxo de Monty Hall, há 3 portas. Atrás de uma há um carro de luxo e atrás das outras duas há cabras. O participante escolhe uma porta (digamos a Porta 1). O apresentador, que sabe onde está o carro, abre uma das outras duas portas (digamos a Porta 3), revelando uma cabra. Ele então pergunta: 'Você deseja trocar para a Porta 2?'. Matematicamente, trocar de porta aumenta as chances de vitória? Qual é a probabilidade exata se ele trocar e se ele mantiver a escolha?",
        "pensamento": """[Entendimento]: O usuário apresenta o clássico problema de probabilidade condicional de Monty Hall.
O participante quer maximizar sua probabilidade de ganhar o carro.
Premissas fundamentais do problema padrão:
1. O carro é colocado aleatoriamente em uma das 3 portas com probabilidade 1/3 para cada uma.
2. O apresentador SEMPRE sabe onde o carro está.
3. O apresentador SEMPRE abre uma porta que tem uma cabra (ele nunca revela o carro).
4. O apresentador SEMPRE oferece a opção de troca.

[Desenvolvimento]:
1. Análise da probabilidade inicial (sem troca):
   - O jogador escolhe a Porta 1.
   - A probabilidade de o carro estar na Porta 1 é P(C1) = 1/3.
   - A probabilidade de o carro NÃO estar na Porta 1 (ou seja, estar no conjunto {Porta 2, Porta 3}) é P(C2 ou C3) = 2/3.

2. A ação informada do apresentador:
   - O apresentador abre a Porta 3 e revela uma cabra.
   - O apresentador NUNCA abre uma porta aleatória; ele usa o conhecimento prévio para filtrar uma cabra dentre as duas portas restantes.
   - Isso significa que toda a probabilidade combinada de 2/3 que estava sobre o conjunto {Porta 2, Porta 3} agora colapsa exclusivamente sobre a Porta 2!
   - Portanto, a probabilidade da Porta 2 conter o carro passa a ser exatamente 2/3.

3. Demonstração por Espaço Amostral Completo (3 cenários equiprováveis de 1/3 cada):
   - Cenário A (Carro na Porta 1): Jogador escolhe P1. Apresentador abre P2 ou P3. Se o jogador TROCAR, ele PERDE. (Vitória mantendo: 1/3).
   - Cenário B (Carro na Porta 2): Jogador escolhe P1. Apresentador é forçado a abrir P3. Se o jogador TROCAR para P2, ele GANHA. (Vitória trocando: +1/3).
   - Cenário C (Carro na Porta 3): Jogador escolhe P1. Apresentador é forçado a abrir P2. Se o jogador TROCAR para P3, ele GANHA. (Vitória trocando: +1/3).

4. Somatório das probabilidades:
   - Se mantiver a porta: ganha apenas no Cenário A -> Probabilidade = 1/3 (≈ 33,33%).
   - Se trocar de porta: ganha nos Cenários B e C -> Probabilidade = 2/3 (≈ 66,67%).

[Auto-Verificação]:
Deixe-me conferir se há falácias comuns:
- Por que a chance não é 50/50? A intuição popular erra ao achar que, por restarem duas portas fechadas, cada uma tem 50%. Isso só seria verdade se o apresentador abrisse uma porta aleatoriamente sem saber o que havia atrás e por pura sorte encontrasse uma cabra (Monty cego). Como o apresentador tem conhecimento deliberado, ele injeta informação no sistema.
- A probabilidade de 2/3 é matematicamente comprovada pelo Teorema de Bayes:
  P(C2 | A3) = [P(A3 | C2) * P(C2)] / P(A3)
  P(A3 | C2) = 1 (se o carro tá na 2 e escolhi 1, Monty só pode abrir a 3).
  P(C2) = 1/3.
  P(A3) = 1/2.
  P(C2 | A3) = (1 * 1/3) / (1/2) = 2/3.
  O Teorema de Bayes confirma com 100% de rigor.

[Síntese]: Trocar de porta dobra as chances de vitória, saltando de 1/3 para 2/3.""",
        "resposta": """**Sim, trocar de porta dobra as chances de ganhar o carro!**

### As probabilidades exatas:
* **Se você MANTIVER a porta escolhida:** a chance de vitória é de **1/3 (~33,33%)**.
* **Se você TROCAR de porta:** a chance de vitória sobe para **2/3 (~66,67%)**.

---

### Por que isso acontece (A lógica matemática):

Quando você escolheu a Porta 1 no início, você tinha apenas **1/3 de chance de ter acertado**, e havia **2/3 de chance de o carro estar nas outras duas portas juntas**.

Como o apresentador **sabe onde o carro está** e é obrigado a abrir uma porta que tem uma cabra, ele faz o "trabalho sujo" por você: ele elimina a opção perdedora. Aqueles **2/3 de probabilidade original** que estavam distribuídos entre as portas 2 e 3 agora se concentram **inteiramente na Porta 2**.

Portanto, a troca não é um jogo de 50/50: a troca transfere para você a probabilidade de 2/3 das outras portas combinadas."""
    },
    # -------------------------------------------------------------------------
    # DIREITO CONSTITUCIONAL E PENAL BRASILEIRO (DESAFIO OAB / SABIÁ-4)
    # -------------------------------------------------------------------------
    {
        "categoria": "direito_constitucional_oab",
        "pergunta": "Quais são as cláusulas pétreas da Constituição Federal de 1988 previstas no Artigo 60, § 4º? Uma Proposta de Emenda à Constituição (PEC) que pretenda instituir a pena de morte para crimes hediondos no Brasil é juridicamente viável?",
        "pensamento": """[Entendimento]: A questão aborda o núcleo rígido da Constituição Federal de 1988 (CF/88):
1. Identificar exaustivamente as quatro cláusulas pétreas do Artigo 60, § 4º.
2. Analisar a viabilidade jurídica de uma Proposta de Emenda Constitucional (PEC) que busca instituir a pena de morte para crimes hediondos no Brasil.

[Desenvolvimento]:
1. As quatro Cláusulas Pétreas (Art. 60, § 4º, incisos I a IV da CF/88):
   - I: A **forma federativa de Estado**;
   - II: O **voto direto, secreto, universal e periódico**;
   - III: A **separação dos Poderes** (Legislativo, Executivo e Judiciário);
   - IV: Os **direitos e garantias individuais**.

2. Análise da viabilidade jurídica da PEC da pena de morte:
   - O direito à vida é a matriz de todos os direitos e garantias individuais (Art. 5º, caput da CF/88).
   - O Artigo 5º, inciso XLVII, alínea 'a' da CF/88 estabelece categoricamente:
     "não haverá penas: a) de morte, salvo em caso de guerra declarada, nos termos do art. 84, XIX;".
   - Sendo o direito à vida e a vedação à pena de morte garantias individuais expressas do Artigo 5º, elas integram o rol do Artigo 60, § 4º, IV (Cláusula Pétrea).
   - O Artigo 60, § 4º, determina expressamente que: *"Não será objeto de deliberação a proposta de emenda tendente a abolir..."*.
   - Uma PEC que tente criar a pena de morte em tempo de paz (mesmo para crimes hediondos) constitui uma emenda "tendente a abolir" um direito/garantia individual.

[Auto-Verificação]:
Deixe-me conferir se há exceções ou pegadinhas jurídicas:
- Existe pena de morte no Brasil atualmente? SIM, mas apenas e exclusivamente em caso de **guerra externa formalmente declarada** pelo Presidente da República com autorização do Congresso Nacional (Art. 84, XIX), regulada pelo Código Penal Militar.
- Pode uma PEC estender a pena de morte para tempos de paz ou para crimes hediondos? **NÃO, de forma alguma.** Trata-se de inconstitucionalidade material flagrante por violação de cláusula pétrea (Art. 60, § 4º, IV).
- Tal PEC sequer pode ser votada (deve ser rejeitada de plano pela CCJ ou barrada via Mandado de Segurança impetrado por parlamentar perante o STF).

[Síntese]: Enumerar as quatro cláusulas e fundamentar a impossibilidade jurídica absoluta da PEC.""",
        "resposta": """### 1. As 4 Cláusulas Pétreas da Constituição de 1988 (Art. 60, § 4º)
A Constituição Federal veda expressamente qualquer deliberação sobre proposta de emenda tendente a abolir:
1. **A forma federativa de Estado** (autonomia de Estados, Municípios e DF);
2. **O voto direto, secreto, universal e periódico**;
3. **A separação dos Poderes** (harmonia e independência entre Executivo, Legislativo e Judiciário);
4. **Os direitos e garantias individuais** (Artigo 5º da CF/88).

---

### 2. Viabilidade Jurídica da Pena de Morte para Crimes Hediondos:
**É totalmente INVIÁVEL (Inconstitucional).**

**Fundamentação Jurídica:**
* O **Artigo 5º, XLVII, "a"** da CF/88 estabelece que *"não haverá penas de morte, salvo em caso de guerra declarada"*.
* Por ser um **direito e garantia individual fundamental**, essa proteção é uma **cláusula pétrea** (Art. 60, § 4º, IV).
* Uma Proposta de Emenda à Constituição (PEC) não tem poder de restringir, revogar ou retroceder em direitos fundamentais já assegurados (Princípio do Não-Retrocesso Social e Vedação ao Retrocesso).
* Portanto, tal proposta sequer pode ser submetida a votação no plenário do Congresso Nacional, sendo eivada de **inconstitucionalidade material insanável**."""
    },
    {
        "categoria": "direito_penal_oab",
        "pergunta": "Qual a diferença entre Dolo Eventual e Culpa Consciente no Direito Penal brasileiro? Apresente um caso prático com acidente de trânsito em alta velocidade.",
        "pensamento": """[Entendimento]: Distinção dogmática clássica entre:
1. Dolo Eventual (Art. 18, I, 2ª parte do Código Penal).
2. Culpa Consciente (Art. 18, II do Código Penal).
Ambos ocorrem frequentemente em crimes de trânsito (embriaguez ao volante, racha, excesso de velocidade) e são cobrados exaustivamente na OAB e em tribunais do júri.

[Desenvolvimento]:
1. Estrutura Psicológica:
   - No **Dolo Eventual**, o agente **prevê** o resultado como possível e, mesmo assim, **assume o risco de produzi-lo** (teoria do assentimento/consentimento). A atitude mental é: *"Não quero diretamente matar, mas se morrer, tanto faz; pouco me importa; não paro por causa disso"*. Há indiferença com o bem jurídico alheio.
   - Na **Culpa Consciente**, o agente também **prevê** o resultado como possível, mas **não o aceita** e **acredita sinceramente** que suas habilidades ou a sorte impedirão a ocorrência do desastre. A atitude mental é: *"Eu sei que é perigoso, mas eu piloto muito bem, nada vai acontecer"*. Há leviandade, mas não indiferença ou aceitação do resultado.

2. Consequência Processual Prática no Trânsito:
   - Dolo Eventual: O crime é doloso contra a vida -> Vai a julgamento pelo **Tribunal do Júri** (homicídio simples ou qualificado, pena de 6 a 20 ou 12 a 30 anos).
   - Culpa Consciente: Crime culposo -> Julgado por **Juiz Singular** (Art. 302 do Código de Trânsito Brasileiro - homicídio culposo na direção de veículo automotor, pena consideravelmente mais branda).

[Auto-Verificação]:
Deixe-me exemplificar de forma cristalina:
- Caso 1 (Culpa Consciente): Motorista experiente e sóbrio ultrapassa o limite em via deserta à noite para não se atrasar, achando convictamente que seu reflexo e freios ABS evitarão qualquer colisão. Ocorre o acidente. É culpa consciente.
- Caso 2 (Dolo Eventual): Dois motoristas embriagados disputam racha em avenida urbana movimentada à luz do dia, furando sinais vermelhos. Ao passarem em alta velocidade, assumem ostensivamente a possibilidade de matar pedestres (fórmula de Frank: "seja como for, eu não paro"). É dolo eventual.

[Síntese]: Expor o critério da indiferença (assentimento) versus a confiança sincera na própria habilidade.""",
        "resposta": """A diferença entre **Dolo Eventual** e **Culpa Consciente** reside na **atitude psicológica do agente** em relação ao resultado previsto:

---

### 1. Dolo Eventual (Art. 18, I, Código Penal)
* **Conceito:** O indivíduo prevê que sua conduta pode matar alguém e **assume o risco**, demonstrando total indiferença para com o resultado.
* **Voz Interior do Agente:** *"Eu sei que posso matar alguém andando a 150 km/h nessa rua movimentada, mas não me importo. Se acontecer, paciência."*
* **Caso Prático:** Dois motoristas alcoolizados decidem disputar um "racha" em plena luz do dia em uma avenida central e movimentada, avançando semáforos vermelhos a 140 km/h. Um pedestre é atropelado na faixa. O condutor agiu com dolo eventual, pois assumiu conscientemente o risco ao criar perigo intolerável. Será julgado pelo **Tribunal do Júri**.

---

### 2. Culpa Consciente (Art. 18, II, Código Penal)
* **Conceito:** O indivíduo prevê que o perigo existe, mas **não aceita o resultado**. Ele acredita sinceramente que sua habilidade técnica, reflexos ou sorte impedirão que o acidente ocorra.
* **Voz Interior do Agente:** *"Eu sei que estou rápido demais, mas eu conheço essa estrada como a palma da minha mão e sou um piloto experiente; nada de ruim vai acontecer."*
* **Caso Prático:** Um motorista sóbrio, com pressa para chegar ao hospital com um medicamento, acelera em uma rodovia com pista molhada confiando nos freios ABS do veículo. Em uma curva, o carro aquaplana e atinge outro veículo. Houve culpa consciente (ele previu o risco, mas acreditou convictamente que conseguiria controlar o carro). Será julgado por um **Juiz Singular** por homicídio culposo na direção (Art. 302 do CTB)."""
    },
    # -------------------------------------------------------------------------
    # ENGENHARIA DE COMPUTAÇÃO E BAIXO NÍVEL
    # -------------------------------------------------------------------------
    {
        "categoria": "ciencia_da_computacao",
        "pergunta": "Como funcionam as estruturas de cache L1, L2 e L3 em processadores modernos como o AMD Ryzen (arquitetura Zen 3/Zen 4), e qual é o impacto de um 'Cache Miss' no pipeline de instruções da CPU?",
        "pensamento": """[Entendimento]: A questão aborda a hierarquia de memória de semicondutores modernos (especificamente AMD Ryzen Zen 3 / Zen 4):
1. A arquitetura física e a função de L1 (L1i e L1d), L2 e L3.
2. Latência comparativa (ciclos de clock).
3. O fenômeno de 'Cache Miss' e o custo de penalidade (pipeline stall) ao buscar dados na DRAM principal.

[Desenvolvimento]:
1. A Hierarquia de Cache (SRAM estática ultrarrápida no die de silício):
   - **Cache L1 (Nível 1):**
     * Dividido em L1I (Instruções) e L1D (Dados).
     * Tipicamente 32 KB + 32 KB por núcleo.
     * Latência: ~3 a 4 ciclos de clock (~0.8 a 1 nanossegundo).
     * Exclusivo por núcleo.
   - **Cache L2 (Nível 2):**
     * Tipicamente 512 KB a 1 MB por núcleo (privativo de cada core nos Ryzen Zen 3/4).
     * Latência: ~12 a 14 ciclos de clock (~3 nanossegundos).
     * Funciona como amortecedor direto entre L1 e L3.
   - **Cache L3 (Nível 3 / Last Level Cache):**
     * Compartilhado entre todos os núcleos de um mesmo CCD (Core Complex Die). No Zen 3 e Zen 4, o L3 é unificado (32 MB a 64 MB ou até 96 MB com 3D V-Cache).
     * Latência: ~35 a 45 ciclos de clock (~10 a 12 nanossegundos).

2. O que acontece em um Cache Miss:
   - A CPU tenta ler um dado em L1. Não está lá (L1 Miss).
   - Tenta em L2. Não está (L2 Miss).
   - Tenta em L3. Não está (L3 Miss).
   - A CPU é forçada a acionar o controlador de memória IMC (*Integrated Memory Controller*) e atravessar o barramento até os pentes de memória DDR4/DDR5.
   - Latência da RAM DDR4/DDR5: ~60 a 80 nanossegundos (~200 a 300 ciclos de clock!).
   - **O Impacto:** O pipeline de execução entra em *Stall* (parada forçada). A Unidade Lógica e Aritmética (ALU) fica ociosa sem fazer cálculos durante centenas de ciclos, esperando o dado chegar da RAM.

[Auto-Verificação]:
Deixe-me conferir se a arquitetura Zen 3 se destaca aqui:
- No Zen 2, o CCX era dividido em 4 núcleos com 16MB cada.
- No Zen 3/Zen 4, a AMD unificou o L3 em um bloco contíguo de 32MB para todos os 8 núcleos do CCD, eliminando a latência de comunicação inter-core. Isso revolucionou o desempenho em jogos e tarefas com dependência pesada de dados!
- A explicação técnica está impecável.

[Síntese]: Explicar a pirâmide de memória e o impacto brutal do cache miss.""",
        "resposta": """A hierarquia de memória em processadores modernos como o **AMD Ryzen (arquiteturas Zen 3 e Zen 4)** existe para resolver o chamado *"Memory Wall"* — a diferença colossal de velocidade entre os núcleos da CPU (que operam a mais de 4.5 GHz) e a memória RAM externa.

---

### 1. A Estrutura dos Níveis de Cache (SRAM dentro do chip):

* **Cache L1 (Nível 1 - O mais veloz do mundo):**
  * É dividido internamente em **L1I** (Instruções) e **L1D** (Dados), com cerca de 32 KB + 32 KB dedicados a cada núcleo.
  * **Latência:** Apenas **3 a 4 ciclos de clock** (~0,8 nanossegundo).
* **Cache L2 (Nível 2):**
  * Também é privativo de cada núcleo (512 KB a 1 MB por core no Zen 3/Zen 4).
  * **Latência:** Cerca de **12 a 14 ciclos de clock** (~3 nanossegundos).
* **Cache L3 (Nível 3 / Unificado):**
  * É compartilhado entre todos os núcleos do mesmo bloco de processamento (*Core Complex Die - CCD*), totalizando 32 MB a 64 MB (ou até 96 MB nos modelos com *3D V-Cache*).
  * **Latência:** Cerca de **35 a 45 ciclos de clock** (~10 nanossegundos).

---

### 2. O Que é o "Cache Miss" e Seu Impacto Fatal no Desempenho:

Quando a CPU precisa executar uma instrução, ela busca o dado sequencialmente:
$$\text{L1} \rightarrow \text{L2} \rightarrow \text{L3} \rightarrow \text{Memória RAM Externa}$$

Quando o dado não está em nenhum dos três níveis de cache, ocorre um **Cache Miss (Falta de Cache)**:

1. **Parada do Pipeline (*Pipeline Stall*):** A CPU precisa acionar o barramento de memória para buscar o dado nos módulos de memória RAM (DDR4 ou DDR5).
2. **Penalidade de Tempo Brutal:** A memória RAM leva de **60 a 90 nanossegundos** (mais de **250 a 300 ciclos de clock**) para entregar a informação.
3. Durante esses 300 ciclos, os circuitos matemáticos (ALUs) da sua CPU ficam **completamente parados (ociosos)**, desperdiçando poder de fogo enquanto aguardam a chegada dos dados.

Por isso, técnicas como código alinhado em memória e a unificação do Cache L3 no Ryzen foram revolucionárias para garantir fluidez extrema."""
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
        for item in NOVOS_EXEMPLOS:
            p_norm = item["pergunta"].strip().lower()
            if p_norm not in perguntas_existentes:
                perguntas_existentes.add(p_norm)
                item["treino_formatado"] = f"### Usuário:\n{item['pergunta']}\n\n### Assistente:\n<think>\n{item['pensamento']}\n</think>\n{item['resposta']}"
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                adicionados += 1

    print("=" * 65)
    print("🚀 BATERIA PESADA DE RACIOCÍNIO R1 INJETADA COM SUCESSO!")
    print(f"📦 Amostras antes: {total_antes:,}")
    print(f"✨ Novas amostras de elite adicionadas: +{adicionados}")
    print(f"💎 Total consolidado no cérebro Synapse: {total_antes + adicionados:,}")
    print("=" * 65)

if __name__ == "__main__":
    main()
