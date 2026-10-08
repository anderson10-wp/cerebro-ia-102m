# Resultado do piloto de linguagem

Execução concluída no Kaggle. Preparados 2.413 exemplos novos revisados ou verificáveis por construção, mais 1.134 de replay. Nenhuma API de professor foi chamada. Os checkpoints atuais e launchers de conversa não foram alterados.

## Medidas

| Categoria | Antes | Depois |
|---|---:|---:|
| Objetivas, total | 8/64 | 18/64 |
| Pares completos, com respostas diferentes | 0/32 | 6/32 |
| Negação no texto | 2/8 | 8/8 |
| Fatos ensinados, em perguntas reformuladas | 0/8 | 5/8 |
| JSON | 1/8 | 3/8 |
| Contexto e localização | 2/8 | 1/8 |
| Cópia | 2/8 | 1/8 |
| Informação presente ou ausente | 1/8 | 0/8 |
| Mudança para minúsculas | 0/8 | 0/8 |
| Aritmética | 0/8 | 0/8 |
| Retenção nos testes antigos | 8/12 | 8/12 |

Não houve perda de questões antes acertadas na bateria antiga de 12 perguntas. Houve perdas em categorias do novo diagnóstico. O critério objetivo previamente definido foi satisfeito, mas não o critério qualitativo de conversa útil. Os pares são sintéticos, alguns compartilham estruturas, e o resultado de negação não comprova compreensão geral da negação. Por exemplo, esta bateria não fez um controle sistemático da ordem dos sujeitos.

## Treino e seleção

O modelo completou três épocas, 666 passos. A validação teve losses 1,628, 1,703 e 1,868. O treino parou após duas avaliações sem melhora. O melhor checkpoint veio da primeira época, não do último passo. O tempo registrado de treino mais avaliação final foi 133 segundos, além da preparação e avaliação inicial.

Checkpoint selecionado SHA-256: `b5344d7969b7be70193229c81cd5510e1f5ec6562ecd7b60df8429fe8e6aab0e`.

## Leitura das oito respostas abertas

1. Saudação: devolveu a pergunta `Você pode me ajudar?`, sem oferecer ajuda relevante.
2. Correção da capital do Brasil: respondeu `Não. A capital do Brasil é a capital do Brasil.`, sem informar Brasília.
3. Água para plantas: apenas repetiu `As plantas precisam de água.`, sem explicar a razão.
4. Primeiro passo para estudar: fez uma pergunta de esclarecimento parcialmente relacionada, sem sugerir o passo pedido.
5. Treino versus avaliação: produziu uma definição circular e inadequada.
6. Resumo sobre Joana perder o ônibus: respondeu `música`, sem relação com o texto.
7. Idade não informada: começou corretamente com `Não`, mas a explicação ficou confusa.
8. Ação no computador: reconheceu não ter ferramenta de execução para executar o comando; não alegou execução externa.

A conversa não apresentou melhora suficiente. A fluência, a aderência a instruções abertas e a compreensão de textos continuam fracas. Algumas respostas objetivas podem falhar por critérios de formato exato, mas casos como trocar o local do caderno pelo local da mochila e resumir um texto com `música` são erros substantivos.

## Decisão

Não promover este candidato nem ampliar automaticamente o treinamento deste currículo. Preservar o modelo próprio e os exemplos/diagnósticos como material de pesquisa. O piloto mostrou aprendizado localizado, não um assistente conversacional funcional.

O próximo conjunto precisa dar maior atenção à variedade de respostas completas, assuntos e formas de instrução. Os exemplos explicativos humanos/revisados deste piloto são poucos diante do componente sintético. A perda de validação subindo após a primeira época também impede recomendar simplesmente mais passagens pelos mesmos dados.

Os testes deste piloto agora são diagnósticos conhecidos. Uma nova intervenção deverá usar outra reserva para avaliar progresso, sem usar estas perguntas como evidência independente novamente.

Relatório bruto: `outputs/language-pilot-result/language-pilot/report.json`. Dados e critérios: `README.md`, `manifest.json`, `train.jsonl`, `dev.jsonl`, `test.jsonl`. Candidatos guardados nos artefatos da execução privada: https://www.kaggle.com/code/andersonluan10ad/cerebro-102m-language-pilot-oct08
