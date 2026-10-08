# Cérebro IA — 102M

Projeto pessoal de modelo de linguagem treinado do zero, pausado em 8 de outubro de 2026. Este repositório preserva código, dados, checkpoints, relatórios e tentativas para uma possível retomada. Clonar não inicia treinamento nem chamadas pagas.

## Estado real

O modelo tem 102.257.280 parâmetros: 16 camadas, dimensão 640, 10 cabeças, MLP 1792, contexto 1024 e vocabulário 32.768. Usa RoPE, RMSNorm, SwiGLU e embeddings compartilhados com a saída. O objetivo era conversar em português e, futuramente, ajudar a planejar experimentos com professores externos. Ainda não conversa de maneira confiável nem treina outro modelo autonomamente.

Foram processadas aproximadamente 5,642 bilhões de posições de tokens entre o pré-treino inicial e a continuação na H100; isso inclui repetição do corpus, não 5,642 bilhões de tokens únicos. Os testes pequenos não demonstram qualidade geral.

Referência local: `outputs/local-run-20261007-212517-0aec1e/candidate/best-model.pt`, SHA256 `1ea8da9fa900f7f2ec1beaf5f34511d449ea695732e19d1f7744d4efb21f9a9e`, 8/12 no teste estreito. Pré-treino H100: `outputs/h100-pretrain/result/best-model.pt`, SHA256 `e80f4077b8125f5aaea8eb4a2344e1b0bfa6bf06c4033f27112a6564ecd31a9c`.

Experimentos Kaggle de SFT amplo, raciocínio e linguagem foram concluídos. Houve ganhos em alguns testes, mas falhas de conversa e regressões impediram promoção automática dos candidatos. Os atalhos e estados antigos apontam para versões diferentes; confira o checkpoint antes de usar.

A preparação de `outputs/dialogue-v1/round-03` ficou **incompleta**: professores geraram exemplos, os primeiros 100 receberam revisão e correções manuais, mas o restante não recebeu revisão completa. A construção final do conjunto, revisão independente e novo piloto Kaggle não foram concluídos. Arquivos intermediários e logs documentam falhas, limites e tentativas. Não execute essa etapa como se já estivesse validada.

## Restaurar arquivos grandes

O código e os arquivos menores estão no Git. Modelos, corpus e outros arquivos grandes estão nos assets da release `arquivo-2026-10-08`, deduplicados por SHA256. O manifesto preserva cada caminho original, inclusive cópias idênticas. Um arquivo `.partial` é um download incompleto preservado como tal.

1. Clone este repositório privado.
2. Com GitHub CLI autenticado, execute `gh release download arquivo-2026-10-08 --repo anderson10-wp/cerebro-ia-102m --dir arquivos-release`.
3. Execute `python restore_artifacts.py arquivos-release`.

O restaurador verifica os hashes antes de recriar os caminhos e não sobrescreve arquivos diferentes. Também é possível baixar os assets pela página da release.

## Retomar com cuidado

Use Python 3.12. O ambiente local era Windows com RX 580 8GB, Ryzen 5600G e 16GB RAM. Dependências principais: PyTorch 2.4.1, torch-directml, tokenizers, pyarrow, huggingface_hub e Kaggle CLI. `requirements-archive.txt` registra as versões encontradas no ambiente local; não é uma garantia de compatibilidade em outras máquinas. Não foi incluído o ambiente virtual.

Após instalar as dependências e restaurar os pesos, um exemplo para testar a referência local é:

```powershell
python outputs/h100-pretrain/chat_terminal.py --checkpoint outputs/local-run-20261007-212517-0aec1e/candidate/best-model.pt --device directml
```

Confira `--help` antes de treinar. Wrappers de professores esperam credenciais DPAPI do usuário Windows: elas foram excluídas e precisam ser configuradas novamente. Nenhuma chave de API está incluída. Não há autorização automática para gastar novos créditos.

Experimentos e pesos adicionais podem permanecer no Kaggle:

- https://www.kaggle.com/code/andersonluan10ad/cerebro-102m-sft-treino-oct08
- https://www.kaggle.com/code/andersonluan10ad/cerebro-102m-reasoning-test-oct08
- https://www.kaggle.com/code/andersonluan10ad/cerebro-102m-language-pilot-oct08
- Dataset privado: https://www.kaggle.com/datasets/andersonluan10ad/cerebro-102m-sft-oct08

O arquivamento cobre os arquivos locais existentes, não pesos remotos que nunca foram baixados. Os links permitem localizar esses resultados. Fontes e licenças dos dados estão nos SOURCE-CARDs e manifestos; não há uma licença única atribuída a todo o material de terceiros.

## Conteúdo excluído e alterações do arquivo

Credenciais (`*.dpapi`, `.env`), caches Python e o ambiente virtual foram excluídos. Caminhos absolutos deste workspace e padrões de credenciais nos textos foram substituídos para portabilidade e privacidade. Modelos e dados binários mantêm seus bytes originais. `archive-manifest.json` registra arquivos preservados, exclusões e hashes.
