# Rodada maior de ajuste supervisionado

Base: checkpoint local `local-run-20261007-212517-0aec1e`, 102.257.280 parâmetros.
Fonte externa: https://huggingface.co/datasets/nicholasKluge/instruct-aira-dataset-v2
Revisão fixa: c65b2bf91938480c0683ec396f4f11ca1a79782a; licença Apache-2.0 declarada pelo autor.

Conversas em português com filtros automáticos de formato, tamanho, duplicatas e alguns padrões problemáticos. Os filtros **não garantem correção factual**. Inclui replay dos exemplos locais e os 128 exemplos técnicos em pares. O curso de treinamento do Hugging Face é referência, não um segundo dataset importado.

Treino e validação separados por hash da pergunta normalizada, agrupando perguntas que diferem apenas em números. As questões locais conhecidas ficam fora do treino; são testes de desenvolvimento, não evidência independente de capacidade ampla. A auditoria externa fica fora da seleção de checkpoint e é avaliada somente no final.

Limites: duas épocas, até 75 minutos no laço de treino, batch global 8, LR 1e-5, FP16 com GradScaler. Usa as GPUs CUDA disponibilizadas pelo Kaggle. Preparação e avaliações ficam fora das 75 minutos; limite externo da sessão: 90 minutos. Sem fallback para treino na CPU. Sem novas chamadas às APIs dos professores.

Salva `last.pt` a cada 250 passos, com pesos, otimizador, scaler, ordem dos exemplos e estados aleatórios. Salva o melhor checkpoint ao fim de cada época. A versão original permanece como melhor se a loss de validação não melhorar. Os checkpoints ficam em `/kaggle/working/brain-sft`; a retenção em interrupções abruptas depende do Kaggle. Uma retomada exige um novo job e código de restauração; não é automática.

O relatório final contém comparação antes/depois nos testes locais e acertos dos pares. Melhorar a loss não autoriza promoção automática. O modelo de conversa local não é alterado.

Consultar:
`powershell -NoProfile -File work/kaggle.ps1 kernels status andersonluan10ad/cerebro-102m-sft-treino-oct08`

Baixar após conclusão:
`powershell -NoProfile -File work/kaggle.ps1 kernels output andersonluan10ad/cerebro-102m-sft-treino-oct08 -p outputs/kaggle-sft-oct08`
