# Cérebro V1 — treinamento próprio desde o início

Modelo decoder Transformer com **102.257.280 parâmetros**, inicializados aleatoriamente. Nenhum peso, embedding ou tokenizer de outro modelo é carregado. O tokenizer BPE de 32.768 entradas é treinado no corpus deste projeto.

**Estado em 5 de outubro de 2026:** o pré-treino concluiu 1.999.962.112 tokens, e os pesos foram baixados em `trained/brain`. Um piloto posterior de instruções no Lightning gerou um candidato separado; ele não passou na avaliação e não substituiu esta base. Os resultados e a revisão estão em `../agent-pilot/README.md` e `../agent-pilot/pilot-review.json`.

## Configuração

| Item | Valor |
|---|---|
| Parâmetros | 102.257.280, contando pesos compartilhados uma vez |
| Blocos Transformer | 16 |
| Dimensão | 640 |
| Cabeças de atenção | 10, dimensão 64 |
| Camada feed-forward | SwiGLU, dimensão 1.792 |
| Normalização e posições | RMSNorm e RoPE |
| Contexto de treinamento | 1.024 tokens |
| Vocabulário | BPE próprio, 32.768 entradas |
| Objetivo | Prever o próximo token |
| GPUs | 2 Tesla T4, treinamento DDP em FP16 |
| Batch global | 65.536 tokens: 8 sequências × 1.024 × 2 GPUs × 4 acumulações |
| Otimizador | AdamW, betas 0,9 e 0,95, weight decay 0,1 em matrizes |
| Taxa de aprendizado | Aquecimento de 20 milhões de tokens; pico 0,0006; decaimento cosseno até 0,00006 |
| Meta | Cerca de 2 bilhões de tokens processados, limitada pela cota |
| Reserva | O controlador preserva pelo menos 3 horas disponíveis e limita o orçamento desta sequência a 26 horas de GPU |

O limite exato de tokens é arredondado para baixo para batches completos: 1.999.962.112 tokens. Não há garantia de atingir a meta se houver falhas ou desempenho inferior ao benchmark.

## Dados e separação

60% português do FineWeb-2, 25% inglês do FineWeb-Edu e 15% Python do CodeParrot Clean. As revisões de origem ficam registradas em `sources.json`. FineWeb-Edu contém textos web selecionados com auxílio de um classificador; isso não envolve carregar pesos desse classificador no nosso modelo.

Os documentos são deduplicados por hash de texto com espaços normalizados. A separação de treino, validação e auditoria ocorre antes do tokenizer; textos de validação e auditoria não treinam o tokenizer. Essa deduplicação exata não garante eliminar documentos quase iguais. O corpus mantém um registro de proveniência e licenças. Os registros de código são filtrados por licenças indicadas em `sources.json`.

O benchmark de GPU usa dados sintéticos apenas para medir desempenho. Os testes de retomada também usam um corpus sintético separado. Esses testes não contam como o treinamento de linguagem.

## Execução

A preparação roda em CPU no notebook privado:

https://www.kaggle.com/code/andersonluan10ad/cerebro-v1-102m-corpus

`orchestrate.py` espera o corpus e os testes de GPU concluírem, confere a cota, cria até quatro sessões privadas de treinamento com retomada do otimizador e baixa os pesos de inferência ao concluir. O acompanhamento é registrado em `run-state.json`. O controlador executa neste computador; o treino que já estiver rodando no Kaggle continua se o computador dormir, mas a próxima sessão e o download dependem de o controlador estar ativo.

Cada sessão salva `last.pt` com modelo, otimizador, scaler, progresso e identidade dos dados; `model.pt` contém os pesos de inferência mais recentes; `best-model.pt` contém a menor perda de validação observada. A auditoria reservada é consultada ao atingir a meta. As notas medem previsão de texto, não inteligência geral ou qualidade de conversa.

Após desligar ou reiniciar o computador, o controlador precisa ser iniciado novamente. Ele recupera `run-state.json`, verifica as sessões já enviadas e continua a partir dos checkpoints, sem reiniciar o treino nem ampliar o orçamento original. Um bloqueio de processo evita executar dois controladores ao mesmo tempo. Da pasta raiz deste chat, execute `python outputs/cerebro-v1/orchestrate.py --workspace .`.

O início de uma sessão faz verificação de tamanho dos arquivos e identidade do tokenizer/manifesto. Os hashes completos do corpus ficam registrados no manifesto, mas não são recalculados em todas as retomadas.

## Usar os pesos depois do treinamento

```powershell
python generate.py --checkpoint trained/brain/best-model.pt --tokenizer trained/brain/tokenizer.json --prompt "A inteligência artificial pode" --tokens 100
```

PyTorch, NumPy, tokenizers, huggingface_hub e PyArrow são dependências do código. Nenhum token do Kaggle é colocado nos arquivos de treinamento. O acesso local usa a credencial protegida por Windows DPAPI já configurada neste chat.

## O que esta versão entrega

Uma base pequena de linguagem, com pesos próprios e treinamento reproduzível. A primeira versão não tem treinamento para seguir instruções, navegação na internet ou capacidade implementada de criar suas sucessoras. Essas funções exigem dados, ferramentas, avaliações e etapas posteriores. O tamanho do modelo e a cota disponível limitam a capacidade obtida.

Os relatórios `gpu-benchmark.json`, `local-verification.json`, `gpu-integration.json` e o relatório final do treinamento documentam o que realmente foi executado. `run-state.json` informa se o projeto está preparando dados, treinando, concluído ou interrompido.
