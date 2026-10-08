# Piloto de linguagem e instruções

Base própria de 102.257.280 parâmetros: checkpoint de referência `local-run-20261007-212517-0aec1e`, SHA-256 `1ea8da9fa900f7f2ec1beaf5f34511d449ea695732e19d1f7744d4efb21f9a9e`.

## Dados preparados

2.413 exemplos novos: compreensão de pequenos textos fictícios, identificação de sujeito e local, negação, informação ausente, resumos, histórico curto de diálogo, cópia, transformação de caixa, JSON, contas pequenas e respostas explicativas revisadas. 1.134 exemplos de replay mantêm as tarefas anteriores. Os textos fictícios são a fonte explícita das respostas; a verificação é por construção, não por reputação de um repositório.

As respostas explicativas foram escritas e revisadas pelo Codex nesta sessão. Isso não é revisão humana independente nem cobre fatos amplamente. O conjunto é um currículo pequeno para diagnosticar aprendizado, não um corpus geral de conversa.

Foi examinada uma amostra de 100 registros de `marlosb/smol-smoltalk-pt`, revisão `fe468af00de1f8d948a40399755e1faae0fa9cda`. Foram encontrados exemplos que descumprem instruções e código traduzido incorretamente (`if` transformado em `se`, entre outros). Apenas duas tarefas de restrições foram adaptadas, com respostas corrigidas; nenhum lote bruto foi incorporado. O autor declara Apache-2.0: https://huggingface.co/datasets/marlosb/smol-smoltalk-pt

## Avaliação e execução

64 questões objetivas reservadas em oito categorias, oito questões abertas para leitura manual e 12 questões antigas de retenção. Há também 64 exemplos de validação para escolher o checkpoint. Os testes foram escritos antes da execução, com verificação de ausência de prompts idênticos no treino. Isso não elimina semelhança entre tarefas nem garante reserva semântica completa. As perguntas abertas sobre conceitos conhecidos avaliam reformulação de assuntos ensinados, não conhecimento adquirido sem exposição.

Inferência greedy, até 48 tokens nas perguntas objetivas e 80 nas abertas. Resposta exata ou JSON com tipos preservados. Esse critério pode rejeitar variantes semanticamente corretas; respostas abertas não entram no total automático. Só pares com respostas distintas entram na métrica de pares completos.

AdamW LR 2e-5, batch global 16, quatro épocas no máximo, limite de 600 segundos no laço de treinamento e 1.200 segundos de sessão. FP16 com GradScaler e CUDA, usando as GPUs disponíveis; sem treino em CPU. Salva melhor e última versão por época. A menor loss de validação escolhe o candidato; a avaliação final mede o comportamento. Sem APIs de professores, retomada automática, promoção automática ou treino maior automático.

Critério objetivo definido previamente: pelo menos oito acertos adicionais nas 64 questões e nenhuma regressão nos 12 testes de retenção. Além disso, as respostas abertas precisam ser lidas manualmente quanto a aderência, correção e repetição antes de recomendar ampliar o treino. Passar no critério objetivo sozinho não basta.

Notebook: https://www.kaggle.com/code/andersonluan10ad/cerebro-102m-language-pilot-oct08

Relatório de conclusão:
`powershell -NoProfile -File work/kaggle.ps1 kernels output andersonluan10ad/cerebro-102m-language-pilot-oct08 -p outputs/language-pilot-result --file-pattern 'report.json$' -q`

`manifest.json` registra hashes dos arquivos de dados. A entrada do notebook contém a mesma carga JSON comprimida para respeitar o limite de tamanho de código do Kaggle; os arquivos locais permanecem legíveis. Não há credenciais no código enviado.
