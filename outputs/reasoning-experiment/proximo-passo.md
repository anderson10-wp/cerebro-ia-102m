# Continuar com o modelo próprio

O projeto continua com o modelo próprio de 102M. O modo de etapas não será ativado por padrão: nesta rodada teve 40/100 acertos, contra 52/100 da cópia direta, e pior desempenho nas formulações novas.

A checagem local da cópia direta terminou: manteve 8/12 no teste básico, sem regressões nessa bateria, e passou de 6/16 para 9/16 nas outras perguntas objetivas. Porém perdeu duas respostas antes corretas: aumento de loss e treino em andamento. Assim, não passou no critério de retenção e não foi promovida. O checkpoint anterior permanece como referência; a cópia direta fica como candidato experimental. Relatório: `retention.json`.

1. Avaliar a retenção da cópia direta nos 12 testes básicos e nas 16 perguntas objetivas do diagnóstico anterior. Não promover se perder habilidades antes acertadas. Manter os checkpoints antigos separados.
2. Priorizar uma tarefa pequena ligada ao objetivo final: interpretar informações de uma execução e escolher uma ação de um vocabulário permitido. A decisão de GPU disponível teve 18/20 no teste sintético, mas exige outra avaliação com descrições variadas, dados ausentes e casos contraditórios antes de uso real.
3. Corrigir essa competência com dados verificáveis e preservar um conjunto de replay. Separar instâncias de treino, validação e avaliação; testar reformulações além de números novos. Considerar um componente de classificação separado sobre o modelo congelado ou um adaptador reversível, para reduzir alterações nas habilidades anteriores. Nenhuma dessas opções já foi implementada neste passo.
4. Somente após esse teste, conectar a saída ao controlador em simulação. O controlador deve conferir a disponibilidade real de GPU e rejeitar comandos fora da lista. O modelo não poderá executar treinamento por conta própria neste estágio.
5. Ampliar para leitura de logs, interpretação de resultados de ferramentas e seleção de configurações, uma competência por vez, exigindo melhora nas perguntas reservadas e retenção dos acertos anteriores.

O objetivo de coordenar treinamentos permanece, mas não há evidência atual de que o modelo consiga desenvolvê-los autonomamente. Contas e verificações objetivas poderão usar ferramentas; o modelo precisa aprender a solicitá-las e interpretar seus resultados. Não há nova rodada longa iniciada neste passo.
