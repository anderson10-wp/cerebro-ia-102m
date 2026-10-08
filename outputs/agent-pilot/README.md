# 102M principal, GPT-OSS como professor

O principal continua sendo o **102.257.280 parâmetros treinado do zero**, com tokenizador próprio e 1.999.962.112 tokens de pré-treino. O GPT-OSS 20B forneceu exemplos de instruções; seus pesos não foram copiados para o principal.

## Resultado real — 5 de outubro de 2026

O job privado `cerebro-102m-instructions-01` terminou no Lightning em uma Tesla T4. Foram 600 passos, 599 atualizações válidas e 95.948 tokens de resposta processados, incluindo repetições. A etapa de treino durou 74,73 segundos; preparação, avaliação e alocação ficam fora desse tempo. O melhor checkpoint por loss de validação foi o **passo 50**; depois a validação piorou.

O custo registrado do job foi **0,09533333 crédito**. O saldo passou de 29,88 antes da preparação para **29,78 créditos**. O Studio está `Stopped`, sem máquina, e o job está `Completed`. A sessão anterior do professor no Kaggle também terminou. Nenhum novo job foi iniciado nesta revisão.

| Avaliação | Original | Candidato: pontuação mecânica | Candidato: revisão de conteúdo |
|---|---:|---:|---:|
| Validação | 0/16 | 8/16 | 8/16 |
| Auditoria | 0/16 | 8/16 | **4/16** |

Na auditoria, 15/16 respostas passaram na validação de estrutura e disponibilidade. Quatro perguntas de esclarecimento contavam como corretas no avaliador mecânico, mas foram rejeitadas na revisão: não perguntavam pelo que faltava ou afirmavam que o professor estava indisponível sem evidência. O resultado mecânico permanece intacto. `pilot-review.json` registra a revisão separadamente.

O candidato encaminha corretamente quatro multiplicações simples da validação e oito consultas ao estado do treino distribuídas entre validação e auditoria. Falha nas quatro expressões com parênteses, nas quatro situações com calculadora indisponível, nas explicações de arquitetura e nos pedidos de desenho de experimentos. Também faz perguntas inadequadas. **Nenhuma ferramenta foi executada.**

A NLL de uma pequena sonda de linguagem com oito textos e 163 tokens subiu de 3,4042 para 3,5677: aproximadamente 4,8%. Isso sugere regressão nessa sonda; não mede linguagem ou conhecimento em geral. O candidato **não passou no critério declarado antes do treino** e não substituiu o original.

## Dados e limites

O GPT-OSS recebeu somente 32 pedidos de treino: 28 respostas aprovadas e quatro corrigidas pelo Codex. Os registros estão em `teacher-routing-results.json` e `data-review.json`.

O treino foi ampliado para **543 exemplos**: 32 sementes revisadas e 511 variações programáticas do protocolo. São 82.288 tokens incluindo prompts e 10.871 tokens de resposta no conjunto, antes das repetições. Esse material é pequeno, repetitivo e focado em escolher entre calculadora, professor, consulta ao treino e esclarecimento. Não representa conversas naturais amplas nem demonstra autonomia.

Validação e auditoria tinham 16 casos cada, congelados antes do ajuste. O checkpoint foi escolhido somente pela loss de validação. Agora que os erros da auditoria orientam melhorias, esse conjunto serve como diagnóstico: uma próxima rodada precisa de **nova auditoria independente**, congelada antes do treino.

## Decisão e próxima etapa

Preservar o original como principal e guardar o candidato do passo 50 para testes. Preparar dados mais diversos com pares de ferramentas disponíveis/indisponíveis, cópia completa de expressões, distinção entre explicações e consultas ao estado real, e perguntas específicas sobre informação faltante. Antes de outro job, congelar uma nova avaliação e usar validação frequente com parada antecipada. O plano está em `pilot-review.json`.

O orçamento reservado foi de até 2 créditos; o restante permanece disponível. Um modelo de 1B do zero e o GPT-OSS 120B continuam adiados. O resultado atual não justifica gastar o saldo inteiro nem permitir treinamentos autônomos.

## Arquivos e reprodução

- `lightning-result/report.json`: treino, seleção e métricas.
- `lightning-result/before.json` e `after.json`: pedidos, saídas e pontuações.
- `lightning-result/best-model.pt`: candidato separado do original.
- `pilot-review.json`: revisão, decisão e plano.
- `candidate-integrity.json`: verificação dos pesos baixados e do original.
- `lightning-pilot-state.json`: job, orçamento, custo e saldo.
- `pilot_train.py`: experimento executado; `sft.py` é o auxiliar e a receita preliminar.
- `pilot-data-manifest.json` e `bundle-manifest.json`: identidade dos dados e do pacote.

Para experimentar o roteamento local, da raiz do projeto:

```powershell
python outputs/agent-pilot/try_candidate.py "Calcule 8 * 7."
python outputs/agent-pilot/try_candidate.py "Calcule 9 + 2." --available teacher training_status clarify
```

O demonstrador usa CPU, gera JSON livremente e verifica estrutura e disponibilidade. **Não verifica toda a semântica nem executa ferramentas.** Uma expressão parcial pode passar na verificação de estrutura e continuar errada. A saída registra `semantic_correctness_verified: false` e `tools_executed: 0`.

Para consultar o job sem iniciar uma máquina:

```powershell
powershell -NoProfile -File outputs/agent-pilot/cloud.ps1 -Mode status
```

Os wrappers usam credenciais criptografadas pelo DPAPI do Windows e restauram as variáveis de ambiente. O pacote usa uma lista explícita de arquivos e não inclui essas credenciais.

O controlador experimental `run.ps1` produziu um plano válido em formato, mas rejeitado na revisão por ganhos e limites de latência sem evidência (`proposal-review.json`). O endpoint do professor é temporário e sua sessão terminou.

