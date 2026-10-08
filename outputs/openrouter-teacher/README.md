# Professor OpenRouter

Chave autenticada e protegida pelo DPAPI do Windows em work/openrouter-api-key.dpapi. Conta identificada como free tier. Integração compartilhada: outputs/gemini-teacher/teacher.py --provider openrouter.

Teste batch-301 concluído: 24 respostas aprovadas, zero rejeitadas. Roteador openrouter/free selecionou apodex/apodex-1.1-mini:free. Uso reportado: 4231 tokens; custo da chamada reportado pela API: 0. Nenhum treino foi iniciado. Os exemplos têm respostas verificáveis automaticamente; isso não demonstra qualidade geral do professor nem melhoria do aluno.

Execute professor-openrouter.cmd na raiz do projeto; próximo lote padrão 302. Para outro lote use -Seed 303. Cada execução faz uma chamada limitada a 4096 tokens de saída. Lotes existentes são recusados; não há repetição automática ou fallback pago. O roteador gratuito pode selecionar modelos diferentes; o modelo efetivo é registrado no relatório e na origem das respostas. Os testes reservados permanecem locais.

Arquivos de saída: train.jsonl, rejected.json e report.json dentro de batch-N. O script não usa Codex e não altera pesos. Exemplos aprovados ainda precisam ser incorporados a um treino com revisão dos anteriores e avaliação.
