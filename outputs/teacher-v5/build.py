"""Broader short-answer curriculum, complete v3 replay, fixed evaluation splits."""
import hashlib,json,shutil
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    tok=Tokenizer.from_file(str(ROOT.parent/'h100-pretrain/result/tokenizer.json'))
    rows=[json.loads(x) for x in (ROOT.parent/'teacher-v3/train.jsonl').read_text(encoding='utf-8').splitlines()]
    replay=len(rows)
    def add(cat,prompt,response,scorer='exact'):
        rows.append(dict(id=f'v5-new-{len(rows)-replay:03}',split='train',category=cat,prompt=f'Pergunta: {prompt}\nResposta:',response=response,scorer=scorer,provenance='Teacher-designed deterministic examples; full v3 replay retained'))
    names=['Bia','Caio','Eva','Davi','Nina','Hugo','Ivo','Sara','Luca','Mila','Otto','Yara','Teo','Cleo','Vera','Noel']
    words=['sapato','montanha','azulejo','estrela','tecido','caderno','árvore','planeta','garrafa','coelho','escada','concha','violino','tijolo','tapete','sino']
    locations=['na estante','no bolso','sobre a mesa','atrás da porta']
    for i,(name,word) in enumerate(zip(names,words)):
        place=locations[i%4]
        prompts=[f'Bilhete: {name} guardou o {word} {place}. Onde ficou o {word}? Só o local.',f'O {word} foi colocado {place} por {name}. Extraia apenas o local.',f'Pessoa: {name}. Objeto: {word}. Local: {place}. Qual é o local?',f'{name} disse: deixei o {word} {place}. Responda apenas onde ele está.']
        add('context',prompts[i%4],place)
        missing=['idade','telefone','cidade','profissão'][i%4]
        absent=[f'Cadastro: nome {name}; {missing} ausente. Informe {missing} ou "não informado".',f'No registro de {name}, não consta {missing}. Responda "não informado" se o dado não aparece.',f'Ficha: nome = {name}. Não há outros dados. Qual é a {missing}? Se não constar, escreva "não informado".',f'{name} não forneceu {missing}. Extraia esse dado; use "não informado" se estiver ausente.']
        add('grounding',absent[i%4],'não informado')
        # Pair missing-data cases with present-data cases: do not teach a constant answer.
        add('context',f'Cadastro: nome {name}; cidade Porto. Informe somente a cidade.','Porto')
        copy=[f'Responda só: {word}',f'Sua resposta deve ser apenas a palavra {word}.',f'Copie sem acrescentar texto: {word}',f'Escreva {word} e pare.']
        add('instruction',copy[i%4],word)
        seq=[word,words[(i+5)%16],words[(i+9)%16]]
        idx=i%3; label=['primeira','segunda','terceira'][idx]
        selection=[f'Escreva a {label} palavra de "{" ".join(seq)}". Só ela.',f'Texto: {" ".join(seq)}. Retorne apenas a palavra na posição {idx+1}.',f'De "{" ".join(seq)}", extraia a {label} palavra.',f'Qual é a {label} palavra? Frase: {" ".join(seq)}. Sem explicação.']
        add('word_selection',selection[i%4],seq[idx])
        key=['pronto','ligado','aberto','visivel'][i%4]; value=bool((i//4)%2)
        add('json',f'Produza somente JSON: chave {key}, valor booleano {"verdadeiro" if value else "falso"}.',json.dumps({key:value},separators=(',',':')),'json')
        n=i%8; values=list(range(n))
        add('code',f'Qual o resultado de len({values}) em Python? Só o número.',str(len(values)))
        a,b=i+2,i%5+1
        operation=['+','-'][i%2]; result=a+b if operation=='+' else a-b
        add('arithmetic',f'Responda apenas o número: {a} {operation} {b}.',str(result))
    # Preserve machine-readable output and extracting supplied tool results.
    for i in range(8):
        val=101+i
        add('tool_result',f'Retorno simulado: {{"resultado":{val}}}. Extraia só o número, sem executar ferramentas.',str(val))
    for split in ['validation','audit']:
        shutil.copyfile(ROOT.parent/f'teacher-v3/{split}.jsonl',ROOT/f'{split}.jsonl')
        held={json.loads(x)['prompt'].strip().casefold() for x in (ROOT/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()}
        assert not held & {r['prompt'].strip().casefold() for r in rows}
    for row in rows:
        row['prompt_tokens']=len(tok.encode(row['prompt']).ids)
        row['response_tokens']=len(tok.encode(row['response']).ids)
        assert row['prompt_tokens']+row['response_tokens']+1<=1024
        if row['scorer']=='json': json.loads(row['response'])
    (ROOT/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    manifest={s:{'sha256':sha(ROOT/f'{s}.jsonl')} for s in ['train','validation','audit']}
    manifest.update(tokenizer_sha256=sha(ROOT.parent/'h100-pretrain/result/tokenizer.json'),training_examples=len(rows),new_examples=len(rows)-replay,replay_examples=replay,exact_prompt_overlap=0,warning='Development-guided curriculum; repeated templates are not independent data. Audit not evaluated.')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    recipe=json.loads((ROOT.parent/'teacher-v4/pilot-config.json').read_text())
    recipe.update(learning_rate=3e-6,max_epochs=2,max_training_seconds=600,retention_policy='Keep every v3 training example; start from v3, not regressed v4')
    recipe['train_sha256']=manifest['train']['sha256']
    (ROOT/'pilot-config.json').write_text(json.dumps(recipe,indent=2),encoding='utf-8')
    shutil.copyfile(ROOT.parent/'teacher-v4/pilot.py',ROOT/'pilot.py')

    print(json.dumps(manifest))
if __name__=='__main__': main()

