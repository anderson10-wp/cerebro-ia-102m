# Primeiro lote: português e operação de treino

Foram baixadas 500 linhas públicas do split portuguese de nicholasKluge/instruct-aira-dataset-v2 pela API de datasets do Hugging Face. Nenhuma API de professor foi chamada. O cache e seu hash permitem rastrear a amostra. O dataset tem respostas sintéticas; foram encontrados erros factuais e de tradução, por isso não houve importação indiscriminada.

16 perguntas foram selecionadas e suas respostas revisadas e reescritas em português curto. Isso é revisão pelo Codex, não certificação independente. A origem por linha, atribuição, licença declarada Apache-2.0 e indicação de modificação constam no JSONL. Autor do conjunto: Nicholas Kluge Corrêa. Fonte: https://huggingface.co/datasets/nicholasKluge/instruct-aira-dataset-v2 . Nenhuma afirmação de endosso do autor.

64 exemplos técnicos originais usam registros simulados de loss, acertos, regressões, limite de tempo e status. A referência pedagógica é https://huggingface.co/learn/llm-course/chapter7/6 . Não são trechos copiados do curso nem demonstrações de escrever um treinador completo. As respostas são calculadas por regras explícitas. 16 casos adicionais permanecem em technical-validation.jsonl, fora do treino; compartilham templates e testam somente transferência estreita. Ainda não foram avaliados no aluno.

Saída integrada: outputs/curriculum-teacher/batch-001/train.jsonl (80 exemplos). O importador já inclui curriculum junto com replay e outros professores, respeitando o limite de 2000 exemplos. Cada exemplo novo cabe em 180 tokens do tokenizador local; nenhum prompt coincide exatamente com os conjuntos de avaliação existentes. Não foi realizado treinamento completo nesta preparação.

Abra treinar-curriculo.cmd na raiz para uma rodada de CPU até 10 minutos, mais verificação e avaliação, sem chamadas externas. Parte do candidato de 6/12 e salva nova versão separada. Não substitui o chat. treinar-automatico.cmd também inclui os novos dados, mas continua pedindo exercícios às APIs conforme sua configuração. O teste atual de 12 perguntas não mede conversa ampla ou habilidade de programar treinos; deve ser ampliado antes de conclusões sobre essas habilidades.

Os demais dados baixados permanecem apenas como cache, sem ingresso automático no treino. Este lote é um piloto de integração, não os 2000 exemplos inicialmente sugeridos.

Atualização: os comandos treinar-ia.cmd, treinar-automatico.cmd e treinar-curriculo.cmd agora usam a RX 580 via DirectML no ambiente .venv-directml. O treinamento cancela se a placa não for encontrada; não muda silenciosamente para treino CPU. A CPU ainda prepara dados, executa algumas operações de apoio não suportadas pelo DirectML e a avaliação de respostas, limitada a 2 threads. Conversar continua com sua configuração anterior.
