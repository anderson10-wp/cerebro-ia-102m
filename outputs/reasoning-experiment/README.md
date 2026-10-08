# Teste de respostas diretas versus etapas verificáveis

O experimento parte do checkpoint local anterior à rodada maior do Kaggle, SHA-256 `1ea8da9fa900f7f2ec1beaf5f34511d449ea695732e19d1f7744d4efb21f9a9e`. Essa versão teve 6/16 acertos no diagnóstico anterior, contra 5/16 na rodada maior. Nenhum modelo existente é substituído.

Cinco famílias: redução de loss, orçamento de tempo, promoção de candidato, disponibilidade de GPU e adição. São 1.000 exemplos novos de treino em pares, 50 de validação e 100 de teste. Metade dos testes usa uma formulação nova; todos usam instâncias reservadas. As perguntas de GPU diferem apenas em identificadores de execução e disponibilidade, portanto avaliam uma regra simples, não raciocínio complexo.

As duas cópias começam nos mesmos pesos e recebem o mesmo histórico de 1.134 exemplos, os mesmos casos novos, três épocas, batch 16, AdamW com LR 1e-5 e semente 21911. Cada braço tem limite de 600 segundos. O número de passos pretendido é igual; o braço com etapas contém mais tokens de resposta, portanto não tem custo computacional exatamente igual. A melhor versão de cada braço é selecionada pela loss de sua própria validação. As losses dos braços não devem ser comparadas diretamente, porque as respostas têm comprimentos e conteúdos diferentes.

Quatro condições: base respondendo diretamente; base solicitada a escrever uma etapa; cópia treinada para responder diretamente; cópia treinada para escrever uma comparação e uma resposta final.

O resultado é medido por resposta final exata, pares com ambas as condições corretas e, no modo com etapas, comparação verificada mais resposta final correta. O verificador aceita apenas o fato explícito esperado na linha `Comparação:`. Uma explicação alternativa correta pode falhar nesse critério estrito. Ele não transforma nem corrige as respostas do modelo.

Esses testes são sintéticos e pequenos, não comprovam capacidade de conversar, desenvolver modelos ou controlar treinamentos. Não há APIs de professores nem promoção automática. Uma decisão de promover exige testes de retenção adicionais.

Notebook privado: https://www.kaggle.com/code/andersonluan10ad/cerebro-102m-reasoning-test-oct08

Status:
`powershell -NoProfile -File work/kaggle.ps1 kernels status andersonluan10ad/cerebro-102m-reasoning-test-oct08`

Relatório:
`powershell -NoProfile -File work/kaggle.ps1 kernels output andersonluan10ad/cerebro-102m-reasoning-test-oct08 -p outputs/reasoning-result --file-pattern 'report.json$' -q`
