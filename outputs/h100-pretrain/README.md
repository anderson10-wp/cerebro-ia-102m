# Continuação experimental na H100

Orçamento desta primeira rodada: aproximadamente 27 créditos, conforme autorização posterior para usar o saldo disponível e preservar uma pequena margem. Modelo próprio: 102.257.280 parâmetros. Ponto de partida: checkpoint completo do terceiro treino no Kaggle, com 1.999.962.112 tokens processados e estado do AdamW.

## Receita

- Uma H100 de 80 GB; BF16 com pesos e estado do otimizador em FP32.
- Atenção SDPA Flash; comparação de microbatches 8, 16, 32 e 64, sem atualizar pesos durante a medição.
- Lote global preservado: 64 sequências de 1.024 tokens por atualização.
- Continuação por até 16.200 segundos, com limite externo de 16.800 segundos e uma única tentativa do job.
- Checkpoints a cada aproximadamente 180 segundos; interrupção após duas avaliações consecutivas com regressão acima de 0,01 na loss ponderada de validação.
- O modelo original fica preservado. Resultados novos só devem ser adotados após avaliação.

O corpus contém 2 bilhões de tokens de treino, já vistos no treino anterior. Tokens adicionais desta rodada representam novas passagens pelo mesmo corpus, não dados únicos novos. Contar mais tokens, por si só, não comprova melhoria.

## Verificação

`recipe-verification.json` registra o teste em CPU da retomada do otimizador e da equivalência entre um lote completo e acumulação de gradientes. A validação de identidade dos dados usa tamanho e SHA-256 de cada arquivo. O conjunto de auditoria não participa da escolha do checkpoint.

## Estado real

Consultar `cloud-state.json` para IDs e estado do recurso. `prepared` significa dados prontos, não treino iniciado. Só logs do job com eventos `train` confirmam atualizações do modelo na H100. `result/report.json`, quando existir, registra tokens adicionais, configuração escolhida e validação.

Dados são transferidos primeiro em CPU; a GPU não é solicitada durante esse preparo. Credenciais e URLs assinadas não devem aparecer nos artefatos nem nos logs.
