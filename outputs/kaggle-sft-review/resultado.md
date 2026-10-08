# Verificação após a rodada Kaggle

Inferência local na RX 580, sem treinamento e sem chamadas às APIs. Checkpoint baixado e SHA-256 conferido contra o relatório da sessão Kaggle. Ambos os modelos receberam exatamente as mesmas perguntas e geração greedy. Respostas objetivas limitadas a 24 tokens; abertas a 64 tokens.

| Medida | Versão anterior | Nova versão |
|---|---:|---:|
| Questões objetivas com resposta exata | 6/16 | 5/16 |
| Pares com ambas as condições corretas | 0/5 | 0/5 |

Foram também examinadas quatro respostas abertas por modelo. Esses números vêm de um conjunto pequeno de diagnóstico, não de um benchmark amplo. As perguntas foram verificadas contra sobreposição exata com os dados desta rodada. Algumas tarefas são reformulações de famílias já usadas; após este diagnóstico, deixam de ser testes intocados.

## Exemplos da nova versão

- Copiar `marfim`: acertou. Copiar `azul`: respondeu `azulejo`. A versão anterior acertou os três testes de cópia, a nova acertou dois.
- `23 + 18`: respondeu `21`; `7 vezes 6`: respondeu `12`.
- JSON com booleano false: acertou.
- Loss de 1.50 para 1.32: respondeu `não` à pergunta se houve redução.
- GPU disponível: respondeu `tREINAR`, falhando no formato exato. GPU indisponível: também produziu texto com `TREINAR`, falhando na decisão necessária.
- Saudação simples: respondeu sobre encontrar um plano de saúde, sem isso ter sido solicitado.
- Explicação de log: não explicou os campos nem a necessidade de uma referência anterior.
- Pergunta sobre suas ações no Kaggle: descreveu incorretamente o Kaggle como um aplicativo de mensagens, sem esclarecer suas capacidades.

## Interpretação

O ganho anterior de 28/60 para 41/60 e 0/16 para 8/16 pares ocorreu nos testes locais conhecidos da rodada. As novas formulações não preservaram esse ganho. A loss melhorou no dataset de conversas, mas isso não demonstrou compreensão robusta de instruções e decisões técnicas.

A regressão original usava `Sua saída deve conter só: cobre`. O novo teste `Repita apenas a palavra cobre.` foi acertado; isso não prova que a regressão original foi corrigida. O modelo continua sensível à formulação.

Não promover automaticamente a nova versão e não iniciar outra rodada longa apenas com mais do mesmo dataset. Próxima intervenção recomendada: currículo técnico mais variado, com respostas curtas verificáveis e pares de condições opostas; reduzir o peso de conversas abertas neste estágio. A avaliação seguinte deverá incluir outra reserva inédita de formulações.

Arquivos brutos: `cases.jsonl`, `before.json`, `after.json`. O checkpoint novo permanece em `outputs/kaggle-sft-oct08/brain-sft/best-model.pt`. Nenhum launcher existente ou modelo atual de conversa foi substituído.
