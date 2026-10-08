# Analisador de experimentos

Abra analisar-ia.cmd na raiz do projeto. Faz UMA chamada Groq GPT-OSS 120B para analisar o candidato 212517 e suas avaliações geral e técnica. Não dispara treino, executa comandos sugeridos, muda checkpoints nem promove o modelo. Sem Codex na execução local. Custo/cota da Groq se aplica; não há repetição automática.

Cada execução grava evidence.json, proposal.json, validated-proposal.json, report.json e PLANO.txt quando a proposta passa nas verificações. Diretório: outputs/experiment-analysis/DATA-ID. proposal.json preserva a resposta original; validated-proposal normaliza somente o alias conhecido RX 580 DirectML para directml.

São verificados campos, referências de evidência, dados em pares, separação de avaliação, replay, ausência de uso do audit reservado, limite de 600 segundos, batch até 4, learning rate entre 1e-6 e 3e-6 e nenhuma execução/promoção automática. Formato válido não prova qualidade científica; a proposta ainda precisa ser implementada e testada. O professor vê um resumo compacto e oito respostas técnicas, nunca as chaves ou o checkpoint. DPAPI segue o wrapper existente.

Verificação sem API: Python outputs/experiment_analyzer.py. Testes offline: Python outputs/test_experiment_analyzer.py. Usar Python314 instalado, como no atalho. O programa verifica os hashes das duas avaliações contra o checkpoint antes de enviar os dados. Esta primeira versão analisa explicitamente o candidato de 8/12, não procura ou promove automaticamente qualquer arquivo novo.

Teste real de 08/10/2026: uma chamada, 1774 tokens totais reportados. Proposta: 128 exemplos pareados, 32 casos separados, experimento futuro de 300 segundos, batch 2, learning rate 2e-6. A resposta inicialmente usou o nome de exibição da GPU; o alias foi tratado localmente e a mesma resposta reaproveitada sem outra chamada. Treinamento não iniciado.
