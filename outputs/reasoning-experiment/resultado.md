# Resultado do experimento de etapas verificáveis

O experimento terminou. As duas cópias partiram dos mesmos pesos, receberam os mesmos 1.000 exercícios e 1.134 exemplos de replay, e treinaram por três épocas, 402 passos cada, em duas T4. Cada braço levou cerca de 73 segundos de treino. Não houve chamadas às APIs dos professores nem promoção automática.

| Medida reservada | Treinada para resposta direta | Treinada com comparação antes da resposta |
|---|---:|---:|
| Respostas finais corretas | 52/100 | 40/100 |
| Pares com ambas as condições corretas | 11/50 | 5/50 |
| Perguntas com formulação nova | 23/50 | 14/50 |
| Comparação verificada e resposta correta | Não se aplica | 29/100 |

Pedir etapas ao checkpoint original, sem o ajuste, não gerou respostas válidas no formato exigido. Esse resultado mistura compreensão e adesão ao formato; não deve ser interpretado como ausência de qualquer habilidade no modelo original.

Na cópia com etapas, para loss de 4.7 para 4.1, a saída foi `Comparação: 4.1 > 4.7. Final: não`, embora o valor tenha caído. Para loss de 8.4 para 9.1, a saída final `não` estava certa, mas veio acompanhada da comparação falsa `8.4 > 8.4`. Há erros reais nos fatos intermediários, além de possíveis falhas de formato no verificador estrito.

A queda da loss de validação foi forte, mas não se traduziu em melhora nas perguntas reservadas. As losses dos braços não são comparáveis entre si, pois os alvos têm conteúdos diferentes. O braço com etapas recebeu mais tokens supervisionados; o controle igualou exemplos, épocas e passos, não FLOPs ou quantidade de tokens.

Conclusão: neste ajuste curto, o modo com etapas piorou os acertos. Isso não prova que todo treinamento de raciocínio falharia; mostra que este currículo e esta rodada não justificam ativar o modo nem trocar o modelo atual. Até a cópia direta segue fraca em aritmética e em mudar respostas conforme condições. Testes adicionais de retenção seriam necessários para qualquer promoção.

Relatório bruto baixado: `outputs/reasoning-result/reasoning-test/report.json`. Checkpoints separados permanecem como artefatos no Kaggle. Nenhum launcher ou checkpoint atual de conversa foi alterado.

Execução: https://www.kaggle.com/code/andersonluan10ad/cerebro-102m-reasoning-test-oct08
