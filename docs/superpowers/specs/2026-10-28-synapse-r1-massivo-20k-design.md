# Design Document: Synapse-R1 Massivo 20k — Expansão Cognitiva de Fronteira

**Data:** 28 de Outubro de 2026  
**Autor:** Nandinho (Engenheiro Chefe) & Antigravity (Parceiro de Engenharia IA)  
**Status:** Revisado e Aprovado pelo Reviewer Técnico  

---

## 1. Visão Geral e Intenção

### Contexto
O projeto **Synapse** desenvolveu do zero uma arquitetura **Sparse MoE (1.21B parâmetros totais, 261M ativos por token, 12 especialistas SwiGLU, Top-2 routing, RoPE, 20 camadas, dim=768)**, pré-treinada em 327 milhões de tokens com checkpoint base `synapse-1b-step-10000.pt`.
A primeira prova de conceito de alinhamento com raciocínio profundo (DeepSeek-R1 style) utilizou 2.825 amostras e confirmou que a arquitetura aprende de forma emergente o modo pensante (`<think>`), estruturando respostas com auto-verificação, rejeição ativa de alucinações e prova real reversa.

### Intenção (O "Porquê")
Modelos sovereign e de fronteira como Sabiá-4 (Maritaca AI) e DeepSeek-R1 devem sua alta capacidade lógica não apenas ao tamanho, mas à **densidade e volume de exemplos de raciocínio multi-passo**. Para elevar a Synapse a um patamar competitivo contra modelos de fronteira em olimpíadas de matemática, programação em Python, dedução lógica e soberania de conhecimento nacional, este projeto expandirá o corpus de raciocínio supervisionado para **20.000 a 25.000 amostras de alta densidade**, seguido por um novo ciclo de alinhamento cognitivo na GPU A100.

---

## 2. Metas, Não-Metas e Anti-Metas

### Metas (Goals)
1. **Escala de Raciocínio (20.000 - 25.000 Amostras):** Compilar, filtrar e validar um master dataset (`dataset_synapse_r1_20k.jsonl`) com prompts em português, pensamentos `<think>` estruturados em 4 fases e respostas precisas.
2. **Distribuição Multidisciplinar Calibrada (Anti-Colapso):**
   - **35% Matemática & Raciocínio Quantitativo:** Álgebra, aritmética com prova real, problemas de olimpíadas (OBMEP, Canguru, GSM8k).
   - **30% Algoritmos & Programação:** Python fluente, depuração de erros, simulação mental de execução de código, estruturas de dados.
   - **20% Lógica Formal, Dedução & Pegadinhas:** Problemas de raciocínio estrito, auto-correção de premissas falsas embutidas nas perguntas do usuário, rejeição ativa de alucinações.
   - **10% Conhecimento Soberano Brasileiro & Ciência:** Direito (OAB, princípios constitucionais, penal/civil), história, geografia e ciência nacional.
   - **3% Identidade Soberana da Synapse:** Identidade clara (criada por Nandinho, MoE 12x Top-2), empatia e postura didática (calibrada para evitar repetições excessivas).
   - **2% Respostas Diretas / Saudações:** Instruções cotidianas sem `<think>` para manter a naturalidade e evitar sobre-pensamento em "olá" ou "bom dia".
3. **Pipeline 100% Gratuito e Offline (Zero Custo de API):**
   - Extração e ETL local a partir de grandes acervos abertos de raciocínio no Hugging Face (ex: subconjuntos em português do Bespoke-Stratos, NuminaMath-CoT, OpenHermes-PT e Alpaca-GPT4-PT).
   - Sem dependência de chaves de API externas pagas ou riscos de bloqueio por quota de tokens.
4. **Treinamento Cognitivo de Alta Eficiência na GPU A100:**
   - Treinar via PyTorch com AMP BFloat16 no Google Colab.
   - Máscara de loss `-100` estrita (o modelo só é penalizado pelo `<think>` e pela resposta, nunca pelo prompt do usuário).
   - Otimizador AdamW com Cosine Learning Rate Schedule (LR máx: 3e-5 -> mín: 3e-6, warmup de 3%).
5. **Validação de Integridade de Sequência:**
   - Filtro rígido de contexto: **nenhuma amostra truncada**. Todas as 20k amostras devem caber estritamente dentro de 1024 tokens (`len(tokens) <= 1024`), garantindo que o fechamento `</think>` e a resposta final completa estejam sempre presentes.

### Não-Metas (Non-Goals)
- Não re-treinaremos o pré-treino base do zero (o checkpoint `synapse-1b-step-10000.pt` é o alicerce fundamental).
- Não faremos chamadas a APIs pagas durante a inferência local (a Synapse continuará 100% autônoma e gratuita).
- Não alteraremos os hiperparâmetros fundamentais da arquitetura neural (`dim=768`, 20 camadas, 12 experts), preservando total compatibilidade com o checkpoint de 10.000 passos.

### Anti-Metas (Anti-Goals)
- **Anti-Truncamento de Pensamento:** Jamais cortar uma amostra no meio do `<think>` ou da resposta durante o treino. Se passar de 1024 tokens, ela é compactada ou descartada no ETL.
- **Anti-Slop:** Nenhuma amostra no dataset terá pensamentos genéricos ou vazios (ex: "Pensando na resposta..."). Todas devem conter rigor analítico (`[Entendimento]`, `[Busca/Desenvolvimento]`, `[Auto-Verificação]`, `[Síntese]`).
- **Anti-Perda de Identidade:** O modelo jamais deve afirmar que é da OpenAI, Anthropic, Google ou Meta. Ele é invariavelmente a Synapse, concebida por Nandinho.
- **Anti-Catastrophic Forgetting:** O alinhamento não pode degradar a fluência em português comum que o modelo aprendeu no pré-treinamento.

---

## 3. Abordagens Consideradas

| Abordagem | Descrição | Prós | Contras | Decisão |
| :--- | :--- | :--- | :--- | :--- |
| **Abordagem A: Geração via API de terceiros (OpenRouter / GPT-4o)** | Fazer 20.000 requisições via API para gerar respostas do zero. | Automação simples de script. | Custo financeiro proibitivo (~$50-$100+), risco constante de bloqueio de chave/cota e latência de dias. | **Rejeitada** |
| **Abordagem B: Criação 100% manual amostra por amostra** | Escrever à mão 20.000 exemplos de raciocínio. | Qualidade máxima e controle absoluto. | Inviável no tempo (levaria meses de trabalho ininterrupto). | **Rejeitada** |
| **Abordagem C: ETL e Fusão Híbrida Open-Source (Escolhida)** | Extrair os melhores subconjuntos abertos de raciocínio de alta dificuldade (Bespoke-Stratos, NuminaMath, GSM8k, HumanEval), normalizar em PT-BR no formato de 4 fases R1, fundir com o acervo calibrado nacional e identidade de Nandinho com filtro rigoroso de 1024 tokens. | **Custo R$ 0,00**, velocidade de execução imediata, rigor matemático, português fluente, soberania temática e garantia de integridade total. | **Aprovada pelo Engenheiro Chefe** |

---

## 4. Arquitetura e Engenharia de Implementação

### 4.1 Estrutura do Dataset (`dataset_synapse_r1_20k.jsonl`)
Cada linha será um objeto JSON estritamente validado:
```json
{
  "pergunta": "Texto limpo da solicitação do usuário",
  "pensamento": "[Entendimento]: ...\n\n[Desenvolvimento / Cálculo]: ...\n\n[Auto-Verificação]: ...\n\n[Síntese]: ...",
  "resposta": "Resposta final clara, polida e formatada com markdown"
}
```

### 4.2 Pipeline de Compilação
1. **Script de Ingestão e Curadoria:** `compilar_dataset_r1_20k.py`
   - Ingestão dos subconjuntos de raciocínio (matemática, algoritmos, lógica, direito brasileiro).
   - Filtro de tokenização real BPE (`len(tokens) <= 1024`).
   - Normalização para o padrão 4 fases: `[Entendimento]`, `[Desenvolvimento]`, `[Auto-Verificação]`, `[Síntese]`.
   - Injeção das amostras de identidade calibrada (Synapse / Nandinho) e 2% de conversas diretas anti-forfetting.
2. **Auditoria Estatística:** `auditar_dataset_20k.py`
   - Validação dos 20.000 itens (distribuição por domínio, verificação de tags fechadas `</think>`, contagem de tokens).

### 4.3 Treinamento Cognitivo no Colab (`alinhar_synapse_1b_r1_massivo.py`)
- **Base:** Checkpoint `synapse-1b-step-10000.pt` (Google Drive).
- **Batching:** Micro-batch size 2 ou 4 com gradient accumulation de 8 ou 16 (batch efetivo de 32 a 64 sequências).
- **Otimizador:** AdamW ($\beta_1=0.9, \beta_2=0.95, \text{eps}=10^{-8}, \text{weight\_decay}=0.1$).
- **Regime de Épocas:** 3 épocas completas sobre os 20k exemplos (estimativa: ~25 a 35 minutos na A100).
- **Saída:** `synapse-1b-r1-v2-final.pt` (salvo automaticamente no Google Drive e baixado localmente).

### 4.4 Validação e Inferência Local
- Utilização do script nativo acelerado `chat_cerebro_moe.py` e atalho `Cerebro_IA_Chat.bat` na GPU AMD RX 580 (DirectML com fast-path de 2 especialistas por token).
- Bateria de testes de raciocínio comparativa pré/pós treino.

---

## 5. Decisões Técnicas do Construtor (Builder's Calls)
1. **Zero Custo de API:** O pipeline será 100% autônomo baseado em ETL local de bases abertas, sem gastar 1 centavo com APIs de terceiros.
2. **Filtro Estrito de 1024 Tokens:** Nenhuma amostra será truncada. Amostras longas serão descartadas durante a compilação, assegurando que o modelo nunca aprenda pensamentos cortados.
3. **Calibragem de Identidade e Conversa:** 3% para identidade e 2% para saudações diretas, prevenindo colapso de persona e sobre-pensamento em mensagens simples.
