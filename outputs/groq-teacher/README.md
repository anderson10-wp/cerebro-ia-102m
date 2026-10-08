# Professor Groq

Modelo configurado: openai/gpt-oss-120b, confirmado pela lista de modelos autenticada. Segredo protegido com DPAPI do usuário Windows em work/groq-api-key.dpapi, fora do código.

Teste concluído: batch-202, 24 respostas aprovadas por gabarito determinístico e nenhuma rejeitada. Uso informado: 1115 tokens de entrada, 1232 de saída (incluindo 864 de raciocínio), 2347 no total. Nenhum treinamento iniciado; pesos originais intactos. Isso confirma a integração para exercícios simples, não a qualidade geral de um professor nem melhora do aluno.

Execute professor-groq.cmd na raiz do projeto. Por padrão usa seed 203. Para outro lote: professor-groq.cmd -Seed 204. Lotes existentes são recusados antes da chamada. Cada execução tem no máximo uma chamada e 4096 tokens de saída; não há repetição automática. Arquivos gerados: train.jsonl, rejected.json, report.json. O plano/custo Groq não foi identificado; não há garantia de gratuidade. O script não utiliza Codex.

Código e verificadores compartilhados em outputs/gemini-teacher/teacher.py. Testes reservados não são enviados ao professor. As respostas aceitas estão em formato compatível com o treino local, mas devem ser incorporadas com revisão dos exemplos anteriores e avaliação do candidato antes de trocar o modelo.

Referência da API: https://console.groq.com/docs/api-reference
