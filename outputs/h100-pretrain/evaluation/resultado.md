# Comparação do modelo antes e depois da H100

O checkpoint novo foi baixado e executado localmente em CPU, sem iniciar GPU paga nem alterar pesos. Ambos têm 102.257.280 parâmetros e o mesmo tokenizer, verificado por SHA-256. O checkpoint antigo acumula 1.999.962.112 tokens; o novo, 5.642.190.848.

## Método

Dez prompts iguais foram aplicados a ambos os modelos com seed 42, 48 tokens gerados, temperatura 0,7 e top-k 40. Os cinco primeiros reproduzem o teste anterior. Quatro prompts adicionais de completação usaram 32 tokens e top-k 1, para testar o modelo fora do formato de pergunta e resposta. Os resultados completos, os parâmetros de geração e os hashes dos checkpoints estão em `comparison.json` e `completions.json`.

Este é um teste exploratório pequeno. Não comprova ganhos ou perdas gerais, nem mede todas as capacidades do modelo. Código gerado não foi executado. Sintaxe Python válida, isoladamente, não significa que a função esteja correta; saída truncada pode conter uma função adicional incompleta.

## Resultado observado

| Teste | Antes | Depois |
|---|---|---|
| Pergunta sobre a localização do Brasil | Responde África Ocidental, incorreto | Diz fronteira entre Brasil e Paraguai, incorreto |
| Saudação | Desvia para um escritório | Também desvia para um escritório |
| Pergunta “Quanto é 2 + 2?” | Resposta sem relação | Resposta sem relação |
| Completar `2 + 2 =` | Começa com 0.5 | Começa com 2 |
| Responder somente “azul” | Não segue | Não segue |
| JSON com `acao: consultar_status` | Não produz JSON válido | Não produz JSON válido |
| Planejamento antes de treino pago | Sem plano útil | Sem plano útil |
| Saber se consegue treinar outra IA sozinho | Desvia para jogos | Afirma capacidade sem base e desvia; não é um relato confiável de capacidades |
| Completar “A capital da França é” | Começa corretamente com Paris | Repete “a capital da França” |
| Completar “O Brasil fica na América do” | Começa corretamente com Sul | Começa corretamente com Sul; o restante da continuação ainda requer checagem |
| Completar `def add(a, b):` seguido de `return` | Começa com `a.add(a)` | Começa corretamente com `a + b` |
| Completar `def soma(a, b):` | Função semanticamente incorreta | Função semanticamente incorreta |

Na explicação genérica sobre IA, o modelo novo produz texto legível, mas isso não estabelece acerto factual ou seguimento de instruções. Na pergunta em inglês sobre a capital da França, nenhum dos dois respondeu Paris.

## Interpretação

A loss de validação do treino caiu de 2,93825 para 2,86398. Essa melhora na previsão de próximos tokens não garantiu respostas melhores nos testes de conversa. Houve um acerto novo na completação simples de código, mas também repetição em um prompt factual antes respondido corretamente.

O modelo novo ainda não está pronto para controlar ferramentas, avaliar suas próprias capacidades ou conduzir treinamento autônomo. A afirmação textual de que consegue treinar outra IA não demonstra essa habilidade. Para aproximá-lo de um assistente, a próxima etapa recomendada é ajuste supervisionado com exemplos de instruções, respostas corretas e chamadas de ferramentas, seguido de avaliações separadas. Mais passagens pelo mesmo corpus não substituem esse trabalho.

O checkpoint original permanece preservado; esta avaliação não promoveu o novo modelo automaticamente nem iniciou outro treino.
