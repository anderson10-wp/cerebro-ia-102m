"""Targeted short-answer practice plus stratified replay; fixed development/holdout sets."""
import hashlib,json,random,shutil
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    tok=Tokenizer.from_file(str(ROOT.parent/'h100-pretrain/result/tokenizer.json'))
    old=[json.loads(x) for x in (ROOT.parent/'teacher-v3/train.jsonl').read_text(encoding='utf-8').splitlines()]
    groups={}
    for row in old: groups.setdefault(row['category'],[]).append(row)
    rng=random.Random(104); replay=[]
    for rows in groups.values():
        rng.shuffle(rows); replay.extend(rows[:6])
    new=[]
    def add(category,prompt,response,scorer='exact'):
        new.append(dict(id=f'v4-new-{len(new):03}',split='train',category=category,prompt=f'Pergunta: {prompt}\nResposta:',response=response,scorer=scorer,provenance='Deterministic teacher-designed exercises; answers derived from fields or arithmetic',source_kind='targeted_generated_v4'))
    words=['pedra','nuvem','barco','trigo','folha','ponte','vidro','areia','pasta','papel','vento','lago']
    for i,w in enumerate(words):
        add('instruction',f'Escreva apenas esta palavra, sem explicação: {w}',w)
        seq=[w,words[(i+3)%12],words[(i+7)%12]]
        for idx,label in [(0,'primeira'),(1,'segunda')]:
            add('word_selection',f'Qual é a {label} palavra de "{" ".join(seq)}"? Responda só com ela.',seq[idx])
        place=['armário','bolsa','prateleira','mochila'][i%4]
        add('context',f'Registro: objeto = {w}; local = {place}. Informe somente o local.',place)
        add('json',f'Retorne somente um JSON com a chave "{w}" e o valor booleano {"verdadeiro" if i%2 else "falso"}.',json.dumps({w:bool(i%2)},ensure_ascii=False,separators=(',',':')),'json')
        n=i+2
        add('code',f'Em Python, qual é o resultado de len({list(range(n))})? Responda só o número.',str(n))
        a,b=i+3,2*i+5
        add('arithmetic',f'Calcule {a} + {b}. Escreva somente o resultado.',str(a+b))
        add('arithmetic',f'Calcule {a} * 3. Escreva somente o resultado.',str(a*3))
    rows=replay+new
    held=[]
    for split in ['validation','audit']:
        shutil.copyfile(ROOT.parent/f'teacher-v3/{split}.jsonl',ROOT/f'{split}.jsonl')
        held.extend(json.loads(x)['prompt'].strip().casefold() for x in (ROOT/f'{split}.jsonl').read_text(encoding='utf-8').splitlines())
    assert not set(held)&{x['prompt'].strip().casefold() for x in rows}
    for row in rows:
        row['prompt_tokens']=len(tok.encode(row['prompt']).ids)
        row['response_tokens']=len(tok.encode(row['response']).ids)
        if row['scorer']=='json': json.loads(row['response'])
    (ROOT/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    manifest={s:{'sha256':sha(ROOT/f'{s}.jsonl')} for s in ['train','validation','audit']}
    manifest.update(tokenizer_sha256=sha(ROOT.parent/'h100-pretrain/result/tokenizer.json'),training_examples=len(rows),new_examples=len(new),replay_examples=len(replay),exact_prompt_overlap=0,holdout_evaluated=False,warning='Development tasks inform this curriculum; exact prompt separation does not imply independent generalization.')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    recipe=json.loads((ROOT.parent/'teacher-v3/pilot-config.json').read_text())
    recipe.update(base_checkpoint='../teacher-v3/candidate/best-model.pt',base_checkpoint_sha256=sha(ROOT.parent/'teacher-v3/candidate/best-model.pt'),train_sha256=manifest['train']['sha256'],validation_sha256=manifest['validation']['sha256'],learning_rate=5e-6,max_epochs=2,max_training_seconds=600)
    (ROOT/'pilot-config.json').write_text(json.dumps(recipe,indent=2),encoding='utf-8')
    print(json.dumps(manifest))
if __name__=='__main__': main()
