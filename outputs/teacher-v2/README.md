# Primeira etapa de ensino

Objetivo: ensinar respostas curtas, instruções básicas e reconhecimento de limites ao checkpoint próprio de 102M após a H100. Este pacote prepara o próximo experimento; criar exemplos e executar testes não altera os pesos do modelo.

## Material produzido

- `train.jsonl`: 64 exemplos escritos por Codex, em português, cobrindo instruções, leitura de contexto, aritmética, diálogo, limites, código, JSON e ferramentas simuladas.
- `validation.jsonl`: 12 casos distintos para acompanhar o desenvolvimento.
- `audit.jsonl`: 12 casos reservados para avaliação final; não usados na medição inicial do modelo.
- `manifest.json`: origem, contagens, hashes e limitações do conjunto.
- `baseline.json`: respostas reais do checkpoint atual na validação, com pontuação estrita.
- `recipe-validation.json`: compatibilidade do modelo e verificação da supervisão, sem passos do otimizador.

Os 64 exemplos contêm somente 467 tokens de resposta, incluindo fim de sequência. É uma coleção inicial para testar a receita, não um conjunto amplo capaz de garantir um assistente competente. Parte dos exemplos de conta e JSON usa variações programáticas identificáveis no gerador; isso não equivale a dezenas de tarefas independentes. As anotações foram criadas e revisadas pelo próprio Codex, sem revisão humana independente.

## Exemplos de ensino

**Pedido:** Transforme casa em letras maiúsculas. Responda só o resultado.

**Resposta:** CASA

**Pedido:** Texto: O trem saiu de Recife e chegou a Natal. Qual foi o destino?

**Resposta:** Natal

**Pedido:** Texto: A loja abre às 9h. Qual o horário de fechamento? Se o texto não disser, responda não informado.

**Resposta:** não informado

**Pedido:** Você consegue iniciar uma GPU nesta conversa? Nenhuma ferramenta foi conectada.

**Resposta:** Não. Nesta conversa, posso gerar texto, mas não iniciar uma GPU.

## Avaliação e próxima experiência

Resultado inicial observado: **0/12 na validação**, antes de qualquer ajuste. As 12 saídas foram inspecionadas: houve repetição, desvio da pergunta e falha no formato solicitado; não foi apenas uma divergência pequena de pontuação ou de redação. A auditoria permanece sem execução. Esse resultado é a referência para medir a próxima versão, não uma avaliação de todos os conhecimentos do modelo.

O formato é `Pergunta: ...\nResposta:`, consistente com o teste no terminal. No ajuste supervisionado, os tokens do pedido ficam mascarados; a primeira palavra da resposta e o token de fim são supervisionados. Isso foi verificado com um forward e um backward reais no checkpoint atual, sem atualização de pesos.

A avaliação usa geração livre e determinística, até 48 tokens, sem corrigir a saída, completar JSON ou limitar sua gramática. Pontuação exata avalia a resposta inteira; JSON é comparado por conteúdo, preservando tipos. O teste de encaminhamento é uma simulação limitada, não demonstra autonomia real nem autoriza executar comandos gerados.

O próximo piloto deve usar uma cópia do checkpoint H100 e salvar um candidato separado. Antes de escalar o experimento, ampliar e revisar este material, mantendo os testes reservados fora do treino. Selecionar a versão por validação; só depois executar a auditoria. Alvos iniciais de engenharia: pelo menos 9/12 em validação e auditoria, revisão manual dos acertos e ausência de regressões graves nos testes anteriores. Esses limiares não são prova estatística de confiabilidade nem autorização para ferramentas pagas.

Respostas semanticamente corretas com redação diferente podem ser rejeitadas pelo critério estrito. Revisar esses casos separadamente, preservando o resultado automático original. Se a auditoria influenciar correções futuras do conjunto, ela passa a ser diagnóstica e deverá ser substituída antes de uma nova avaliação final.

Não há modo de pensamento longo nesta etapa. Primeiro medimos acerto e obediência a instruções curtas. Raciocínio em etapas será uma experiência posterior com operações verificáveis.

## Reprodução local

Use o Python 3.14 instalado em `C:\Users\Fernando\AppData\Local\Programs\Python\Python314\python.exe`.

1. `prepare.py`: reconstrói os arquivos determinísticos. Não alterar o gerador depois de congelar uma rodada sem criar nova versão.
2. `validate_recipe.py`: confere máscaras, fim de resposta, hashes e gradientes finitos.
3. `evaluate.py`: mede a validação. A auditoria exige selecionar explicitamente `--split audit` e deve ser reservada para o candidato final.

Nenhuma API paga ou GPU de nuvem é usada por esses scripts.
