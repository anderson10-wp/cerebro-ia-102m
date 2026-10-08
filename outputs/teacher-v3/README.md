# Material ampliado e piloto preparado

O conjunto de ensino passou de **64 para 256 exemplos**, distribuídos em **15 categorias**. A receita do piloto foi validada no checkpoint H100 de 102M, com 5,64 bilhões de tokens acumulados de pré-treino. Após autorização posterior, o piloto local foi executado: o candidato passou de **0/12 para 4/12** na validação e não foi promovido. O modelo do chat foi preservado. Veja `resultado-piloto.md`.

## Conteúdo

| Origem | Exemplos |
|---|---:|
| Material inicial preservado | 64 |
| Novos exemplos escritos por Codex | 128 |
| Novos exemplos derivados de regras verificáveis | 64 |
| Total para o piloto | 256 |

São 8.409 tokens de entrada e resposta no total; 1.851 tokens de resposta supervisionada, incluindo o fim de sequência. É material pequeno para um experimento controlado de seguimento de instruções, não um corpus amplo de assistente. As variações aritméticas e estruturadas não contam como tarefas independentes; a origem está registrada em cada exemplo.

Há tarefas de instruções, leitura de contexto, limites, diálogo, código, aritmética, JSON, ferramentas simuladas, resultados de ferramentas, esclarecimento, respostas apoiadas nas informações fornecidas, classificação, edição e transformação de texto. As ferramentas continuam simuladas e nenhuma resposta gerada é executada.

Exemplos do material novo:

- “A entrega foi adiada de segunda para quarta. Em qual dia será a entrega?” → “quarta”.
- “Traduza para mim.” → pergunta pelo texto e idioma de destino.
- “O log diz apenas solicitação recebida. O treinamento já começou?” → informa que esse log não confirma o início.
- Uma função `eh_par(n)` → `return n % 2 == 0`, conferida com entradas pares e ímpares.
- Conversão de um literal Python para JSON → preserva valores e tipos, incluindo booleanos, listas e `null`.

## Revisão realizada

Passaram **122 verificações automáticas**, cobrindo **106 exemplos distintos**: resultados de contas, transformações de texto, JSON, conversão de valores e comportamento de funções simples. As funções foram conferidas por um interpretador restrito de AST, sem executar código arbitrário. Os demais exemplos receberam revisão do próprio Codex; não houve revisão humana independente. Verificar a sintaxe de um JSON não prova, isoladamente, que ele atende ao pedido.

Os 12 casos de validação e os 12 de auditoria da versão anterior foram preservados byte por byte. Não há pedidos duplicados entre treino e testes após normalização. A triagem por similaridade de caracteres não encontrou pares acima do limiar escolhido de 0,85. Isso não garante independência semântica: os conjuntos compartilham habilidades e alguns formatos de saída intencionalmente.

A referência continua **0/12** na validação do checkpoint atual, registrada em `../teacher-v2/baseline.json`. Não foi necessário gerar essas respostas novamente, pois o modelo e os casos são os mesmos. A auditoria continua sem execução.

## Receita do piloto

- Iniciar de uma cópia do checkpoint H100, identificado pelo SHA-256.
- CPU como padrão; o script não inicia máquinas nem acessa serviços de nuvem.
- Lote de 4 exemplos, taxa de aprendizado inicial de 0,00001 e até 3 épocas, totalizando no máximo 192 passos.
- Limite de 10 minutos para o ciclo de treino, além de carregamento, avaliação e salvamento. Esse temporizador não é um limite de cobrança de nuvem.
- Supervisão apenas da resposta e do token de fim; os tokens do pedido ficam mascarados.
- Avaliar a cada época e guardar um candidato separado quando a loss de validação melhorar. Encerrar após duas avaliações sem melhora.
- Não promover automaticamente o candidato nem substituir o checkpoint usado no chat.

A execução padrão de `pilot.py` passou por um forward e backward reais com gradientes finitos, sem passos do otimizador. Posteriormente, o treinamento completo também foi executado em CPU por três épocas; os resultados estão nos arquivos do candidato e da avaliação.

Após o piloto, avaliar a geração do candidato nos mesmos 12 casos de validação. A meta inicial é pelo menos 9/12, mais revisão das respostas e comparação com os testes anteriores de conversa e código. Se passar, executar a auditoria uma única vez, buscando pelo menos 9/12 e revisando os resultados. Esses limiares são critérios experimentais, não garantia de confiabilidade ou autorização para usar ferramentas reais. Se falhar, preservar o modelo atual e analisar os erros antes de aumentar o gasto.

## Arquivos

- `lessons.py`: os 128 novos exemplos escritos.
- `build.py`: geração determinística da versão ampliada.
- `train.jsonl`: material final de treino.
- `manifest.json`: contagens e hashes.
- `split-review.json`: triagem de sobreposição com testes.
- `answer-checks.json`: verificações das respostas.
- `pilot-config.json`: limites e identificação do checkpoint.
- `pilot.py`: validação por padrão; treino somente com `--train`.
- `recipe-validation.json`: resultado real da verificação da receita.

Para executar scripts deste pacote, use o Python em `C:\Users\Fernando\AppData\Local\Programs\Python\Python314\python.exe`. O Python 3.12 padrão deste computador não tem as dependências usadas aqui.

Esta etapa consumiu somente processamento local. Não utilizou API paga, Lightning ou Kaggle. O material ensina respostas curtas; raciocínio longo não foi acrescentado.
