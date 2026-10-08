# Auditoria inicial do modelo próprio

Auditoria sem treinamento, sem novas sessões de GPU na nuvem e sem APIs de professores. Inferência local na RX 580. O objetivo foi investigar a base de linguagem e problemas óbvios de identidade ou alinhamento; não é uma revisão completa do corpus nem prova de ausência de bugs.

## Verificações realizadas

- Os três tokenizers locais de corpus, Kaggle original e H100 têm SHA-256 idêntico.
- Cinco textos com português, acentos, negação, números e JSON fizeram round-trip exato, sem tokens desconhecidos.
- Os checkpoints carregaram com a configuração esperada, 102.257.280 parâmetros e pesos finitos.
- Alterar o último token não alterou os logits anteriores no teste de causalidade da implementação DirectML.
- A construção de alvos dos exemplos de instruções supervisiona o primeiro token da resposta e o EOS, com o prompt mascarado e comprimentos alinhados.
- Revisão dos scripts: pré-treino de próximo token com corpus separado por documento, SDPA causal, identidades de dados verificadas na retomada H100. Esses controles não medem correção factual ou capacidade de conversar.

## Achados concretos

O corpus registrado tem 2 bilhões de posições de tokens de treino: 1,2 bilhão em português, 500 milhões em inglês e 300 milhões em código. Isso não comprova 2 bilhões de tokens textualmente únicos; a deduplicação é exata por documento. O treino inicial processou cerca de 2 bilhões. A H100 processou mais 3,642 bilhões sobre o mesmo corpus, totalizando 5,642 bilhões de tokens vistos, não um corpus novo de 5,642 bilhões.

O pré-treino trabalhou com português de páginas web, inglês educacional e código. O conjunto local de referência contém 1.134 exemplos de instruções; 1.062 têm no máximo quatro segmentos separados por espaços na resposta. Esse indicador também conta JSON e código sem espaços; não é uma medida perfeita de comprimento ou qualidade. A receita local usa poucos exemplos e saídas frequentemente curtas. A rodada maior de conversas acrescentou diversidade, mas seus filtros não garantiram correção factual e o diagnóstico posterior foi pior em novas formulações.

No checkpoint da H100 anterior aos ajustes de instruções, a geração greedy já produziu:

- Para `O Brasil é um país da América do Sul. Sua capital é`: continuação dizendo São Paulo, com repetição. A capital correta é Brasília.
- Para `As plantas precisam de água e luz porque`: começou dizendo que não precisam de água e desviou para água no corpo humano.
- Para uma saudação no formato Pergunta/Resposta: repetição da própria pergunta sem ajuda útil.

O checkpoint atual de referência continuou com os dois primeiros erros e deu uma resposta curta inadequada à saudação. Portanto, os erros não surgiram todos na última rodada de conversas ou no teste de etapas. A base de linguagem e fatos já tinha limitações demonstráveis nestes exemplos.

São apenas quatro prompts por checkpoint, com geração greedy de até 48 tokens. Outras configurações de geração podem reduzir repetições; isso não garante corrigir os fatos ou o entendimento. Não foi feita uma comparação controlada com um modelo externo de 200M.

## Consequência para o projeto

Não atribuir o resultado somente ao número de parâmetros. As verificações iniciais não identificaram tokenizer incompatível ou erro óbvio de deslocamento de labels; a investigação deve continuar com qualidade e variedade de dados e uma avaliação mais ampla da base.

Preservar o modelo próprio. Antes de ensinar controle autônomo de treinamento, fortalecer a etapa de linguagem e instruções simples com dados selecionados e respostas verificadas. A habilidade de classificar comandos pode continuar como experimento isolado, mas seus acertos não substituem a necessidade de uma base que entenda perguntas reformuladas.

Não iniciar outra sessão longa sobre os mesmos dados apenas para aumentar a contagem de tokens. Próxima preparação: amostra de dados de melhor qualidade, benchmark fixo de linguagem e instruções e um piloto medido. Separar fluência, correção factual, compreensão e uso de comandos nas métricas. Só ampliar o treinamento se o piloto mostrar melhora nas formulações reservadas e retenção das habilidades anteriores.

Relatório reproduzível: `report.json`; script: `audit.py`. Nenhum checkpoint existente foi alterado.
