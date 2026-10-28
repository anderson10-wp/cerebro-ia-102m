# Synapse-R1 Massivo 20k Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compilar um corpus de 20.000 a 25.000 amostras de raciocínio profundo `<think>` multidisciplinar em português (matemática olímpica, algoritmos Python, lógica formal, direito brasileiro e identidade sovereign), validá-lo e construir o pipeline de treinamento de alinhamento cognitivo massivo para a GPU A100 no Google Colab.

**Architecture:** Pipeline ETL local de extração e fusão sem custos de API a partir de grandes acervos de raciocínio abertos, normalizados na sintaxe R1 de 4 fases (`[Entendimento]`, `[Desenvolvimento]`, `[Auto-Verificação]`, `[Síntese]`), filtrados para caber estritamente dentro de 1024 tokens BPE. Treinamento supervisionado com mascaramento de loss `-100` no prompt via PyTorch com AMP BFloat16 sobre o checkpoint pré-treinado `synapse-1b-step-10000.pt`.

**Tech Stack:** Python 3.12, PyTorch 2.6+, Hugging Face Tokenizers (BPE 32.768 tokens), JSON Lines, Google Colab A100-SXM4-80GB, AMP BFloat16.

**Spec:** `docs/superpowers/specs/2026-10-28-synapse-r1-massivo-20k-design.md`

## Global Constraints

- O dataset final deve conter entre 20.000 e 25.000 amostras no formato `.jsonl`.
- Custo de API estritamente R$ 0,00 (proibido o uso de APIs pagas de terceiros).
- Comprimento máximo de sequência: estritamente 1024 tokens (proibido truncamento de amostras no meio de `<think>` ou da resposta; amostras longas devem ser descartadas).
- Formato de pensamento estruturado nas 4 fases: `[Entendimento]`, `[Desenvolvimento / Cálculo]`, `[Auto-Verificação]`, `[Síntese]`.
- Calibração de identidade: ~3% de amostras reforçando que a Synapse foi desenvolvida do zero pelo Nandinho em PyTorch com Sparse MoE 12x Top-2.
- Anti-catastrophic forgetting: ~2% de conversas cotidianas diretas sem `<think>`.
- Compatibilidade arquitetural: Preservar os hiperparâmetros exatos de `modelo_synapse_1b.py` (`Config1B(vocab_size=32768, dim=768, layers=20, n_heads=12, hidden_dim=2048, num_experts=12, top_k=2, context=1024)`).

## Review Focus

1. **Amostras truncadas sem tag `</think>`:** O script de tokenização deve rejeitar qualquer amostra que ultrapasse 1024 tokens com EOS, evitando treinar o modelo a deixar pensamentos abertos.
2. **Colapso de repetição de persona:** As perguntas de identidade devem variar em sintaxe, contexto e propósito didático para não condicionar a IA a falar sobre Nandinho em perguntas não relacionadas.
3. **Distribuição de domínio:** Garantir que matemática e código componham pelo menos 60% do total de dados de raciocínio.
4. **Alinhamento de máscaras `-100`:** Garantir que 100% dos tokens do prompt do usuário recebam `-100`, de modo que a loss seja calculada exclusivamente sobre os tokens de `<think>` e da resposta.
5. **Estabilidade de Gradiente no MoE:** O balanceamento de carga dos especialistas (`aux_loss`) deve ser monitorado com coeficiente $0.01$ para evitar colapso de especialistas durante o SFT massivo.

---

### Task 1: Compilador e Normalizador ETL de Raciocínio Massivo (20k+)

**Files:**
- Create: `compilar_dataset_r1_20k.py`
- Test: `test_compilar_dataset.py`

**Interfaces:**
- Consumes: `tokenizer.json`, `dataset_synapse_r1_reasoning.jsonl`, bases abertas de raciocínio (Bespoke-Stratos, NuminaMath, GSM8K, Python code-feedback, OAB/Direito).
- Produces: `dataset_synapse_r1_20k.jsonl` contendo entre 20.000 e 25.000 linhas JSON formatadas com chaves `pergunta`, `pensamento`, `resposta`.

- [ ] **Step 1: Escrever teste de integridade do pipeline ETL**

Criar `test_compilar_dataset.py` validando que cada amostra gerada:
1. Possui chaves `pergunta`, `pensamento`, `resposta` não vazias.
2. Contém as tags de pensamento `[Entendimento]` e `[Auto-Verificação]` no campo `pensamento` (ou pensamento vazio para saudações diretas).
3. Ao ser tokenizada com `tokenizer.json`, o tamanho total `prompt + think + resp + eos` é menor ou igual a 1024 tokens.

- [ ] **Step 2: Executar o teste e verificar falha**

Run: `python -m unittest test_compilar_dataset.py`
Expected: FAIL (módulo ou dataset não existente).

- [ ] **Step 3: Implementar `compilar_dataset_r1_20k.py`**

Implementar script de compilação que:
1. Carrega o tokenizador BPE do projeto (`tokenizer.json`).
2. Faz download / ingestão de subsets abertos de raciocínio (GSM8K traduzido, NuminaMath, Python instruction, Bespoke-Stratos).
3. Transcria e formata o pensamento nas 4 fases DeepSeek-R1.
4. Incorpora o acervo de 2.825 amostras já calibradas de `dataset_synapse_r1_reasoning.jsonl`.
5. Incorpora 600 exemplos variados de identidade da Synapse (Nandinho, MoE 12x Top-2).
6. Incorpora 500 exemplos de respostas diretas (anti-catastrophic forgetting).
7. Filtra rigorosamente por contagem de tokens: descarta qualquer registro cujo total ultrapasse 1024 tokens.
8. Salva em streaming no arquivo `dataset_synapse_r1_20k.jsonl`.

- [ ] **Step 4: Executar a compilação e verificar teste**

Run: `python compilar_dataset_r1_20k.py; python -m unittest test_compilar_dataset.py`
Expected: PASS com arquivo `dataset_synapse_r1_20k.jsonl` gerado contendo >= 20.000 amostras.

- [ ] **Step 5: Git Commit**

```bash
git add compilar_dataset_r1_20k.py test_compilar_dataset.py
git commit -m "feat: adicionar compilador e validador ETL de 20k amostras R1"
```

---

### Task 2: Auditoria Estatística e Distribuição Temática do Corpus

**Files:**
- Create: `auditar_dataset_20k.py`
- Test: `test_auditoria_20k.py`

**Interfaces:**
- Consumes: `dataset_synapse_r1_20k.jsonl`, `tokenizer.json`
- Produces: Relatório estatístico em markdown e console detalhando volume total, distribuição percentual por domínio, tamanho médio de tokens e integridade das máscaras.

- [ ] **Step 1: Escrever teste de auditoria**

Criar `test_auditoria_20k.py` que testa se a função de auditoria rejeita datasets com menos de 20.000 amostras ou com tokens maiores que 1024.

- [ ] **Step 2: Executar teste e verificar falha**

Run: `python -m unittest test_auditoria_20k.py`
Expected: FAIL.

- [ ] **Step 3: Implementar `auditar_dataset_20k.py`**

Implementar rotina de auditoria que:
1. Conta o total exato de amostras.
2. Mede distribuição de comprimento de tokens (mínimo, média, percentil 95, máximo).
3. Classifica as amostras por domínio (Matemática, Código, Lógica, Direito/Brasil, Identidade, Direta).
4. Verifica ausência de caracteres nulos, duplicatas ou strings truncadas.

- [ ] **Step 4: Executar auditoria sobre o dataset gerado**

Run: `python auditar_dataset_20k.py`
Expected: PASS exibindo tabela estatística completa atendendo aos requisitos da spec.

- [ ] **Step 5: Git Commit**

```bash
git add auditar_dataset_20k.py test_auditoria_20k.py
git commit -m "feat: adicionar script de auditoria estatística do dataset 20k"
```

---

### Task 3: Script de Treinamento Cognitivo Acelerado no Colab (A100)

**Files:**
- Create: `alinhar_synapse_1b_r1_massivo.py`
- Test: `test_treino_dry_run.py`

**Interfaces:**
- Consumes: `dataset_synapse_r1_20k.jsonl`, `modelo_synapse_1b.py`, checkpoint `synapse-1b-step-10000.pt`.
- Produces: Checkpoint `synapse-1b-r1-v2-final.pt` (~4.8 GB) no Google Drive.

- [ ] **Step 1: Escrever teste de dry-run de treinamento**

Criar `test_treino_dry_run.py` que instancia um modelo reduzido (2 blocos, batch 2), roda 1 passo de forward, cálculo de loss mascarada `-100`, backward e optimizer step, verificando que o scaler AMP e a loss reduzem sem NaN.

- [ ] **Step 2: Executar teste de dry-run e verificar falha**

Run: `python -m unittest test_treino_dry_run.py`
Expected: FAIL (módulo de treinamento massivo não implementado).

- [ ] **Step 3: Implementar `alinhar_synapse_1b_r1_massivo.py`**

Implementar script de treinamento com:
1. Detecção automática de ambiente Colab e montagem do Google Drive (`/content/drive/MyDrive/`).
2. Localização resiliente do checkpoint `synapse-1b-step-10000.pt`.
3. Carregamento com `weights_only=False` e registro de classes seguras.
4. Suporte a BFloat16 nativo na NVIDIA A100 (`torch.cuda.amp.autocast(dtype=torch.bfloat16)`).
5. Dataset PyTorch `MaskedReasoningDataset` com máscara `-100` precisa no prompt.
6. Gradient accumulation configurado para batch efetivo de 32 ou 64.
7. Otimizador AdamW com CosineAnnealingLR (3 épocas completas, warmup de 3%).
8. Loop de teste interativo pós-treinamento avaliando as 7 perguntas de controle.
9. Salvamento automático com verificação de integridade no Google Drive.

- [ ] **Step 4: Executar dry-run local**

Run: `python -m unittest test_treino_dry_run.py`
Expected: PASS.

- [ ] **Step 5: Git Commit**

```bash
git add alinhar_synapse_1b_r1_massivo.py test_treino_dry_run.py
git commit -m "feat: adicionar script de treinamento massivo de 20k na A100"
```

---

### Task 4: Notebook do Google Colab para Execução com 1 Clique

**Files:**
- Create: `treinar_synapse_1b_r1_massivo.ipynb`

**Interfaces:**
- Consumes: Repositório GitHub, Google Drive montado com `synapse-1b-step-10000.pt`.
- Produces: Treinamento completo na GPU A100 com barra de progresso, logs em tempo real e geração de `synapse-1b-r1-v2-final.pt`.

- [ ] **Step 1: Criar notebook Jupyter estruturado com 4 células essenciais**

1. Célula 1: Montagem do Google Drive e clonagem/atualização do repositório (`git pull origin main`).
2. Célula 2: Verificação da GPU (`!nvidia-smi`) confirmando A100 com 80GB de VRAM.
3. Célula 3: Instalação rápida de dependências (`pip install tokenizers torch`).
4. Célula 4: Execução do treinamento massivo com log visual de épocas e salvamento do modelo final.

- [ ] **Step 2: Verificar sintaxe JSON do arquivo `.ipynb`**

Run: `python -c "import json; json.load(open('treinar_synapse_1b_r1_massivo.ipynb', encoding='utf-8')); print('Notebook válido!')"`
Expected: "Notebook válido!".

- [ ] **Step 3: Git Commit & Push para o GitHub**

```bash
git add treinar_synapse_1b_r1_massivo.ipynb
git commit -m "feat: adicionar notebook do Colab para treinamento massivo 20k R1"
git push origin main
```
