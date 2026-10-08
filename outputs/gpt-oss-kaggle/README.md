# GPT-OSS 20B no Kaggle

Servidor temporário do GPT-OSS 20B original convertido em GGUF MXFP4 pelo projeto ggml-org.

**Estado consultado em 5 de outubro de 2026:** a sessão de inferência está `COMPLETE`; o endpoint salvo anteriormente não deve ser tratado como um serviço ativo. Nenhuma nova sessão foi iniciada nesta revisão.

- Preparação em CPU: `andersonluan10ad/gpt-oss-20b-runtime-prepare`.
- Inferência: `andersonluan10ad/gpt-oss-20b-dual-t4-api`, duas Tesla T4.
- Backend: llama.cpp b11391, CUDA 12.8, divisão por camadas em proporção 1:1.
- Contexto total: 8.192 tokens, uma requisição por vez.
- Sessão limitada a quatro horas desde o início do script de inferência.
- A sessão consome a cota gratuita de GPU do Kaggle mesmo sem receber pedidos.
- Não há chamadas pagas ao OpenAI, Gemini ou OpenRouter.
- O túnel Cloudflare e a sessão Kaggle são temporários; a URL muda a cada execução.
- A API requer autenticação Bearer. A credencial local fica criptografada por Windows DPAPI em `work/gpt-oss-api-key.dpapi`.
- O notebook de inferência é privado; seu código de execução contém a credencial da sessão e não deve ser publicado.
- O modelo só gera respostas. Nenhuma ferramenta ou comando gerado é executado automaticamente.

Depois que `endpoint.json` existir, executar a partir da raiz do projeto:

```powershell
powershell -NoProfile -File outputs/gpt-oss-kaggle/ask.ps1 -Prompt 'Explique o que faz um tokenizer.'
```

Esse cliente envia a autenticação sem imprimir a chave. Ele exibe somente a resposta final, o motivo de término e a contagem de tokens.

Arquivos de origem: `prepare.py` e `serve.py`. O arquivo `serve.py` distribuído nesta pasta contém apenas um marcador de credencial; a versão privada para envio é gerada em `work/gpt-oss-service`.
