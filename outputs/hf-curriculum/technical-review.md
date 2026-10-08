# Avaliação técnica após a rodada 212517

Comparação com o candidato 190808: 4/16 -> 8/16 nos casos técnicos reservados. Ambos avaliados via DirectML, greedy, até 48 tokens, sem alterações dos pesos ou chamadas de API.

O aumento da pontuação não demonstra competência técnica: o novo modelo sempre respondeu "não" nas perguntas de status/loss, "manter" na escolha de versão e "encerrar" no limite de tempo. Cada template tem rótulos equilibrados; uma resposta constante por template já obtém 8/16. Portanto, o candidato está no nível dessa referência trivial e perdeu os quatro acertos do anterior nesses casos.

Na avaliação geral de desenvolvimento, os novos acertos foram copiar "cobre" uma vez e produzir JSON com ativo=false; a pontuação passou de 6/12 para 8/12 sem regressões nesse conjunto. São ganhos de formato, não evidência de que sabe conduzir treinos.

Próximo trabalho: pares de exercícios com uma única condição alterada, tarefas simples de leitura/comparação e testes de consistência entre os pares; desvincular índices e famílias de respostas no gerador. Os casos técnicos agora são desenvolvimento observado, não auditoria inédita. Não iniciar outra repetição idêntica nem promover o modelo a controlador autônomo com base nessas pontuações.
