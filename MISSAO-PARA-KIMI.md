# Missão: transformar a automação de treino em um coordenador útil

Fernando autorizou passar esta missão ao Kimi. Leia este documento, confira os arquivos e continue o trabalho. Não repita as pequenas rodadas de treino que já falharam em demonstrar aprendizado útil. Economize chamadas e tokens. Não exponha credenciais nos relatórios.

## Objetivo e decisão mais recente

O objetivo de longo prazo é manter uma IA própria que coordene ferramentas e outras IAs e, com competência demonstrada, participe do desenvolvimento de próximas versões. O modelo atual de 102M NÃO demonstra essa capacidade. A decisão aceita é pausar o ciclo repetitivo de treino e usar um modelo pronto como coordenador/professor, deixando o 102M como aluno em tarefas estreitas. Não há previsão confiável de quando o aluno ficará competente.

Sua primeira entrega deve ser um analisador local: lê relatórios reais, chama um professor por API para propor um experimento, valida a proposta e grava um plano legível. Não treine automaticamente nesta primeira entrega. Só depois de validar esse componente, conecte a execução limitada ao controlador existente. Preserve o modelo e os checkpoints anteriores. Não inicie nuvem, GPU paga ou chamadas ilimitadas.

## Ambiente

- Projeto: .
- Windows, Ryzen 5600G, 16 GB RAM, RX 580 2048SP 8 GB.
- Python de treino GPU: .venv-directml\Scripts\python.exe. torch-directml 0.2.5.dev240914 / torch 2.4.1. Modelo e dados já estão locais.
- Python CPU: C:\Users\Fernando\AppData\Local\Programs\Python\Python314\python.exe (torch 2.14 CPU).
- Fernando quer usar a RX 580 para treinar. DirectML usa CPU em algumas operações de apoio. Não mude silenciosamente o treino para CPU. Chat/avaliação podem ser CPU; padrão de avaliação tem 2 threads.
- Não use git add/commit abrangente: o repositório Git pode abranger a pasta pessoal inteira.

## Modelo e evidências

- Decoder próprio: 102.257.280 parâmetros, dim 640, 16 camadas, 10 cabeças, contexto 1024, vocabulário 32768, embedding/head compartilhados. Código: outputs/cerebro-v1/model.py e outputs/h100-pretrain/model.py.
- Pré-treino histórico: aproximadamente 2B tokens no Kaggle + 3,642B processados na H100, com reutilização de corpus. Isso não significa 5,642B tokens únicos.
- Original preservado: outputs/h100-pretrain/result/best-model.pt. Tokenizer: outputs/h100-pretrain/result/tokenizer.json.
- Candidato de 6/12: outputs/local-run-20261007-190808-8c59a4/candidate/best-model.pt.
- Candidato mais recente discutido, de 8/12: outputs/local-run-20261007-212517-0aec1e/candidate/best-model.pt. SHA256: 1ea8da9fa900f7f2ec1beaf5f34511d449ea695732e19d1f7744d4efb21f9a9e.
- Essa rodada: 1134 exemplos, GPU DirectML, 373 atualizações, ~601 segundos, 7139 tokens de resposta apresentados. Loss de validação ~0,688 -> 0,583.
- Nas MESMAS 12 questões de desenvolvimento: 6 -> 8, sem regressões. Novos acertos: copiar 'cobre' uma vez e JSON ativo=false. Ainda erra contas, len([]), segunda palavra. Não extrapole para inteligência geral.
- Teste técnico posterior: 4/16 -> 8/16. O novo candidato sempre respondeu 'não', 'manter' ou 'encerrar', independentemente dos valores. Uma resposta constante por template já obtém 8/16. Houve perda dos quatro acertos técnicos anteriores. NÃO usar isso como prova de compreensão de treinamento.
- Evidências completas: outputs/hf-curriculum/technical-before.json, technical-after.json e technical-review.md.
- As 12 questões e os 16 casos técnicos já são desenvolvimento observado. O audit original de 12 casos em outputs/teacher-v2/audit.jsonl ficou reservado; conferir antes de usar. Testes técnicos compartilham templates e são limitados.

## Arquivos de execução

- outputs/train_local.py: combina replay e fontes aprovadas; valida dados; treina; avalia; salva relatório. Aceita --base-run, --result-file, --seconds (2–600), --device cpu|directml (padrão directml), --check, --smoke. Não chama APIs.
- outputs/teacher-v3/pilot.py: implementação compartilhada do treino. Máscara de prompt -100, EOS supervisionado, AdamW, clipping, checkpoints separados, limite de tempo. Wrappers teacher-v5/pilot.py mudam ROOT.
- outputs/h100-pretrain/directml_backend.py: adapta atenção e loss à RX 580. NÃO simplificar a loss sem comparar gradientes: ignore_index do backend e reduction='none' tiveram comportamento incorreto em testes anteriores. A implementação seleciona explicitamente posições supervisionadas.
- outputs/teacher-v2/evaluate.py: greedy, até 48 tokens, sem reparo de saída. Suporta --cases para avaliação adicional e --device directml. JSON distingue booleanos de números.
- outputs/auto_train.py: uma rodada padrão, até três explícitas. Groq professor -> fallback OpenRouter gratuito -> validação -> treino -> avaliação. Para quando não melhora. Avança base somente com mais acertos sem perder anteriores NO TESTE PEQUENO. Isso é insuficiente como gate de competência geral.
- outputs/automation-local/state.json: ponteiro de continuidade. Leia o conteúdo real; NÃO suponha que aponta para o candidato 8/12. Não troque automaticamente apenas pela pontuação antiga.
- Atalhos: treinar-automatico.cmd, treinar-curriculo.cmd, treinar-ia.cmd usam .venv-directml. treinar-curriculo parte fixamente do candidato 6/12. Não executar outra rodada idêntica agora.
- Locks: outputs/local-training.lock e outputs/automation-local/active.lock. Conferir processos antes de remover um lock; não matar tarefas alheias.
- conversar.cmd usa original H100. conversar-candidato.cmd usa outputs/candidato-atual/best-model.pt, que na última checagem ainda era um candidato 5/12. Não confundir atalho com melhor candidato discutido.

## Professores e credenciais

- Chaves protegidas por DPAPI do usuário Windows em work/groq-api-key.dpapi, work/openrouter-api-key.dpapi e work/gemini-api-key.dpapi. Não copiá-las para código, prompts, Git ou mensagens.
- Usar wrappers outputs/groq-teacher/teacher.ps1 e outputs/openrouter-teacher/teacher.ps1 como referência para carregar/restaurar variáveis de ambiente. Importam Microsoft.PowerShell.Security pelo caminho do PSHOME para evitar erro entre versões PowerShell.
- Integração compartilhada: outputs/gemini-teacher/teacher.py --provider groq|openrouter|gemini.
- Groq: openai/gpt-oss-120b funcionou. OpenRouter: openrouter/free funcionou, modelo efetivo variável e registrado, chamada de teste custou zero. Gemini autenticou mas chamadas retornaram indisponibilidade; não insistir sem necessidade.
- Os professores atuais apenas respondem 24 exercícios simples gerados por regras. Isso NÃO é um planejador de experimentos. Não finja que já implementam a nova missão.
- Limite por chamada existente: 4096 tokens de saída; uma tentativa por provedor. Cota/custo Groq desconhecidos. Sem fallback pago OpenRouter.

## Dados

- Replay teacher-v5; dados programáticos em outputs/programmatic-teacher; respostas validadas em groq-teacher/openrouter-teacher.
- outputs/curriculum-teacher/batch-001: 16 perguntas Aira com respostas revisadas/reescritas + 64 cenários técnicos originais.
- outputs/hf-curriculum contém 500 linhas Aira em cache, fontes, licença Apache-2.0 declarada, atribuição e scripts. Apenas 16 foram incorporadas: amostra bruta tinha erros factuais e de tradução. Não importar tudo automaticamente.
- Fonte português: https://huggingface.co/datasets/nicholasKluge/instruct-aira-dataset-v2
- Referência pedagógica: https://huggingface.co/learn/llm-course/chapter7/6
- Outros repositórios pesquisados, não integrados: huggingface/smollm, huggingface/alignment-handbook, huggingface/smolagents; nanoGPT apenas didático e descontinuado. Não são datasets prontos que ensinem autonomia só por despejar código no treino.

## Entrega concreta esperada

1. Ler métricas e respostas reais do candidato 8/12 e do teste técnico. Produzir um resumo compacto sem enviar checkpoints, arquivos pessoais ou chaves a APIs.
2. Pedir ao professor UMA proposta JSON: problema observado, evidência, hipótese, alteração limitada de dados/configuração, teste novo e independente, orçamento e condição de sucesso. Preferir uma chamada Groq; fallback gratuito só se falhar.
3. Validar o JSON por esquema e regras locais. Apenas ações predefinidas; sem eval, execução de código livre ou comandos arbitrários vindos do professor. O professor propõe, o controlador verifica.
4. Salvar proposta e relatório legíveis, incluindo modelo efetivo, uso, motivo de rejeição se houver. Uma resposta bem formatada não prova que a proposta é boa.
5. Testar caminho válido, proposta inválida, falha de API e limites sem gastar muitas chamadas. Não iniciar treino nesta primeira fase. Explicar para Fernando como executar com um comando fora do Codex.
6. Depois, conectar um experimento por execução ao treinador e comparar com baseline em tarefas novas, incluindo pares contrafactuais. Preservar checkpoints e impedir aprovação baseada só em loss ou nas mesmas 12 perguntas.

Se você não tiver acesso a arquivos ou terminal neste aplicativo, diga isso claramente e forneça os arquivos/comandos para execução local. Não diga que treinou, alterou ou verificou algo sem resultado real de ferramenta.
