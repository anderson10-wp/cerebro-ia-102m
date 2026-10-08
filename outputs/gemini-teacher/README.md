# Professor Gemini

Integração local para obter respostas de referência do Gemini e conferir cada resposta com um gabarito determinístico. Chave guardada com DPAPI do usuário Windows em work/gemini-api-key.dpapi; não consta no código.

## Executar

Na pasta do projeto: `./professor-gemini.cmd -Seed 110`.
Cada execução faz uma chamada, com 24 exercícios e limite de 4096 tokens de saída. Uma seed já existente é recusada antes da chamada para evitar gastos duplicados. Use outra seed para um novo lote. Não há repetição automática, execução de ferramentas propostas pelo professor, alteração de pesos ou treinamento automático.

O plano/cota Google não foi identificado: não há garantia de gratuidade. O processo local não chama Codex. Exercícios e pedidos são enviados exclusivamente à API oficial Google; testes reservados e credenciais de outros provedores não são enviados.

Saídas por batch: train.jsonl (somente respostas aprovadas), rejected.json e report.json (modelo, status, contagem e uso retornado pelo provedor). Este primeiro currículo cobre contas, seleção/cópia de palavras, contexto, JSON e dados ausentes. Não valida habilidades gerais nem explicações abertas.

## Verificação em 07/10/2026

- Consulta autenticada de modelos funcionou.
- 24 casos e comparação JSON distinguindo booleano de número passaram nos testes locais.
- Gemini 2.5 Flash: 404, indisponível para novos usuários; Google indicou 3.8 Flash.
- Gemini 3.8 Flash: duas tentativas com 503 por alta demanda.
- Gemini 3.5 Flash: uma tentativa com o mesmo 503.
- Nenhum exemplo foi gerado pelo serviço e nenhum treinamento foi iniciado. A integração aguarda uma chamada bem-sucedida para validação ponta a ponta.

Documentação: https://ai.google.dev/gemini-api/docs/get-started
