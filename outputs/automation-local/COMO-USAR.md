# Treinamento automático local

Abra `treinar-automatico.cmd` na pasta do projeto. Ele faz uma rodada inteira sem Codex: professor Groq GPT-OSS 120B, verificação determinística das respostas, inclusão dos exemplos válidos com revisão dos anteriores, treino CPU e avaliação. O fallback é OpenRouter gratuito se a primeira chamada falhar. Não é uma tarefa agendada do Codex; nada fica rodando após a conclusão.

No PowerShell, de qualquer pasta:

```powershell
& ".\treinar-automatico.cmd"
```

Para permitir até três rodadas, acrescente `--rounds 3`. Interrompe antes se não houver ganho na avaliação, se uma etapa falhar ou se ambos os professores estiverem indisponíveis. Ctrl+C interrompe. Mantenha o PC ligado, sem suspensão, e a janela aberta. Fechar o Codex não é necessário para o funcionamento; execute pelo seu PowerShell independente.

Limites por rodada: uma chamada Groq e, somente se necessário, uma OpenRouter; 24 exercícios; 4096 tokens de saída por chamada; treino até duas épocas ou dez minutos, mais tempo de carregamento, verificação e avaliação. Não há garantia de gratuidade da Groq. O script não chama APIs do Codex. Não usa H100, Kaggle ou Lightning.

Ponto de partida: candidato local de 6/12, verificado pelo hash do checkpoint avaliado. Só muda a base da próxima rodada quando há mais acertos sem perder acertos anteriores. O chat NÃO é substituído automaticamente. Checkpoints anteriores ficam preservados. O estado de continuidade fica em `outputs/automation-local/state.json` e os relatórios por sessão em `outputs/automation-local/DATA-ID/status.json`. Treino detalhado e avaliação ficam na pasta `outputs/local-run-*` indicada pelo relatório.

O professor atual responde exercícios simples de formato verificável. Ele não edita código nem escolhe livremente gastos ou treinamentos. O teste de 12 questões é de desenvolvimento, pequeno e conhecido pelo processo de ajuste; melhora nele não comprova inteligência geral. O conjunto de auditoria não é utilizado. O limite de 2000 exemplos interrompe o processo antes de um crescimento indefinido.

`--check` verifica a configuração sem API ou treino. `--smoke` testa uma chamada real e treino de poucos segundos sem atualizar a base de continuidade. O teste rápido não serve para medir melhora de qualidade.

Se um encerramento abrupto deixar arquivos `active.lock` ou `local-training.lock`, confirme que não existe processo de treino ativo antes de removê-los. Os bloqueios evitam duas rodadas simultâneas.

Atualização: os comandos treinar-ia.cmd, treinar-automatico.cmd e treinar-curriculo.cmd agora usam a RX 580 via DirectML no ambiente .venv-directml. O treinamento cancela se a placa não for encontrada; não muda silenciosamente para treino CPU. A CPU ainda prepara dados, executa algumas operações de apoio não suportadas pelo DirectML e a avaliação de respostas, limitada a 2 threads. Conversar continua com sua configuração anterior.
