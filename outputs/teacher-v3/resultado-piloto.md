# Resultado do piloto local

O ajuste supervisionado terminou em aproximadamente 6 minutos, no processador, sem API paga ou GPU de nuvem. Foram 3 épocas, 192 atualizações e 5.553 tokens de resposta processados, contando repetições. O melhor candidato foi o da segunda época, no passo 128; a terceira época piorou a loss de validação. Esse candidato foi salvo separadamente.

| Medida | Modelo anterior | Candidato selecionado |
|---|---:|---:|
| Acertos completos em 12 testes de validação | 0 | 4 |
| Loss das respostas de validação | 5,1232 | 1,3235 |

Os acertos foram: reconhecer uma informação ausente no texto, devolver um JSON literal solicitado, reproduzir uma chamada simulada de consulta de status e extrair o número 91 de um resultado fornecido. A chamada simulada já era apresentada no pedido; isso demonstra reprodução do formato, não planejamento autônomo de ferramentas.

Falhas relevantes: 16 + 27 recebeu a resposta 12; a localização de uma chave foi respondida como caixa em vez de gaveta; o JSON com booleano entrou em repetição; pedidos para selecionar uma única palavra continuaram incorretos. As 12 saídas foram revisadas pelo Codex, sem avaliação humana independente.

**Decisão:** não promover o candidato. Ele melhorou nesta pequena validação, mas ficou abaixo da meta prévia de 9/12. O conjunto de auditoria não foi executado e permanece reservado. Os pesos usados por `conversar.cmd` continuam os mesmos.

O ganho mostra aprendizado parcial de respostas curtas e formatos. Não comprova melhora geral de conversa, aritmética ou uso de ferramentas. A próxima revisão dos dados deve dar prioridade à seleção exata de palavras, extração baseada no contexto, números e tipos JSON. Como os erros de validação já são conhecidos, eles não devem ser copiados para o treino nem apresentados depois como evidência independente.

Arquivos: `candidate/best-model.pt`, `candidate/report.json`, `candidate-validation.json` e `pilot-review.json`. O último registra os hashes, a integridade dos pesos e a preservação do checkpoint original.
