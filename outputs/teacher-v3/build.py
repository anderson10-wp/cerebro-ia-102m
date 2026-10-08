"""Versioned curriculum expansion; preserve v2 evaluation files byte for byte."""
import ast,hashlib,json,operator,re,shutil,unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from tokenizers import Tokenizer
from lessons import LESSONS
ROOT=Path(__file__).resolve().parent; OLD=ROOT.parent/'teacher-v2'
TOKENIZER=ROOT.parent/'h100-pretrain/result/tokenizer.json'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def normalized(s): return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    previous=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    assert digest(OLD/'train.jsonl')==previous['train']['sha256']
    rows=[json.loads(s) for s in (OLD/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in rows: row['source_kind']='inherited_v2'; row['source_version']='teacher-v2'
    additions=[]
    def add(category,request,response,source='teacher_authored',rule=None):
        item=dict(id=f'v3-train-{len(additions):03}',split='train',category=category,
            prompt='Pergunta: '+request+'\nResposta:',response=response,
            scorer='json' if category in ('json','structured_data') else 'exact',
            source_kind=source,provenance='Codex-authored; no independent human review',
            scope='Short response; no tool execution or hidden reasoning trace')
        if rule: item['verification']=rule
        additions.append(item)
    for category,examples in LESSONS.items():
        for request,response in examples: add(category,request,response)
    tables={
      '+':[(4,9),(17,8),(26,15),(31,12),(42,16),(53,24),(65,18),(77,14)],
      '-':[(15,8),(28,11),(43,17),(60,24),(81,36),(92,45),(100,37),(120,52)],
      '*':[(2,9),(4,7),(8,11),(9,6),(11,3),(12,4),(7,13),(6,14)],
      '/':[(28,7),(32,4),(45,9),(48,6),(63,7),(84,12),(90,10),(96,8)]}
    functions={'+':operator.add,'-':operator.sub,'*':operator.mul,'/':operator.floordiv}
    for symbol,pairs in tables.items():
        for i,(a,b) in enumerate(pairs):
            number=functions[symbol](a,b)
            if i%2==0:
                request=f'Devolva somente o resultado numérico de {a} {symbol} {b}.'
            else:
                request={'+':f'Uma caixa tinha {a} peças e recebeu mais {b}. Quantas peças há agora? Responda só o número.',
                    '-':f'Havia {a} bilhetes e {b} foram usados. Quantos sobraram? Responda só o número.',
                    '*':f'São {a} grupos com {b} objetos cada. Quantos objetos ao todo? Responda só o número.',
                    '/':f'Distribua {a} fichas igualmente entre {b} pessoas. Quantas fichas cada uma recebe? Responda só o número.'}[symbol]
            add('arithmetic',request,str(number),'rule_generated',{'kind':'arithmetic','a':a,'b':b,'op':symbol})
    transformations=[
      ('uppercase','ponte'),('lowercase','JARDIM'),('reverse','abcd'),('length','melancia'),
      ('digits','r2t5y7'),('letters','h3j4k5'),('no_spaces','p a z'),('join',['leste','oeste','norte']),
      ('sort_numbers',[12,5,18]),('even',[3,4,7,8]),('odd',[2,5,6,9]),('unique',['uva','pera','uva']),
      ('replace','A mala caiu'),('prefix','projeto'),('suffix','arquivo'),('count',[1,3,1,5,1])]
    for rule,value in transformations:
        if rule=='uppercase': request=f'Escreva {value} em maiúsculas. Retorne só a palavra.'; answer=value.upper()
        elif rule=='lowercase': request=f'Escreva {value} em minúsculas. Retorne só a palavra.'; answer=value.lower()
        elif rule=='reverse': request=f'Inverta a ordem das letras de {value}. Retorne só o resultado.'; answer=value[::-1]
        elif rule=='length': request=f'Quantas letras há em {value}? Retorne só o número.'; answer=str(len(value))
        elif rule=='digits': request=f'Extraia apenas os algarismos de {value}, na mesma ordem.'; answer=''.join(c for c in value if c.isdigit())
        elif rule=='letters': request=f'Extraia apenas as letras de {value}, na mesma ordem.'; answer=''.join(c for c in value if c.isalpha())
        elif rule=='no_spaces': request=f'Junte as letras de "{value}", eliminando espaços.'; answer=value.replace(' ','')
        elif rule=='join': request=f'Una {", ".join(value)} usando uma barra entre as palavras, sem espaços.'; answer='/'.join(value)
        elif rule=='sort_numbers': request=f'Ordene {value} do menor para o maior. Responda no formato de lista.'; answer=str(sorted(value))
        elif rule=='even': request=f'De {value}, selecione só os pares, preservando a ordem. Responda como lista.'; answer=str([x for x in value if x%2==0])
        elif rule=='odd': request=f'De {value}, selecione só os ímpares, preservando a ordem. Responda como lista.'; answer=str([x for x in value if x%2])
        elif rule=='unique': request=f'Remova repetições de {", ".join(value)}, mantendo a primeira ocorrência. Separe por vírgula e espaço.'; answer=', '.join(dict.fromkeys(value))
        elif rule=='replace': request=f'Troque mala por bola em "{value}". Retorne só a frase.'; answer=value.replace('mala','bola')
        elif rule=='prefix': request=f'Copie as três primeiras letras de {value}.'; answer=value[:3]
        elif rule=='suffix': request=f'Copie as três últimas letras de {value}.'; answer=value[-3:]
        else: request=f'Quantas vezes o número 1 aparece em {value}? Só o número.'; answer=str(value.count(1))
        add('transformation',request,answer,'rule_generated',{'kind':rule,'input':value})
    records=[{'peso':6},{'altura':1.8},{'habilitado':True},{'falha':False},{'detalhe':None},
        {'nomes':['Ari','Noa']},{'pessoa':{'apelido':'Zeca'}},{'nota':'bom trabalho'},
        {'inicio':3,'fim':11},{'saldo':-8},{'numeros':[10,30]},{'dados':{}},
        {'versao':'v3'},{'limite':250},{'separador':';'},{'pasta':'rascunhos'}]
    for value in records:
        literal=repr(value)
        add('structured_data',f'Converta este literal Python para um objeto JSON, sem explicação: {literal}',
            json.dumps(value,ensure_ascii=False),'rule_generated',{'kind':'python_literal_to_json','literal':literal})
    assert len(additions)==192
    rows+=additions
    tok=Tokenizer.from_file(str(TOKENIZER)); seen=set()
    for row in rows:
        key=normalized(row['prompt']); assert key not in seen, row['id']; seen.add(key)
        row['prompt_tokens']=len(tok.encode(row['prompt'],add_special_tokens=False).ids)
        row['response_tokens']=len(tok.encode(row['response']+'<|eos|>',add_special_tokens=False).ids)
        assert 0<row['prompt_tokens']+row['response_tokens']<=1024
        assert '<|' not in row['prompt'] and '<|' not in row['response']
        if row['scorer']=='json': assert isinstance(json.loads(row['response']),dict)
    (ROOT/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    overlaps=[]; similarities=[]
    for split in ('validation','audit'):
        assert digest(OLD/(split+'.jsonl'))==previous[split]['sha256']
        shutil.copyfile(OLD/(split+'.jsonl'),ROOT/(split+'.jsonl'))
        heldout=[json.loads(s) for s in (ROOT/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        for case in heldout:
            if normalized(case['prompt']) in seen: overlaps.append(case['id'])
            similarity,closest=max((SequenceMatcher(None,normalized(case['prompt']),normalized(r['prompt'])).ratio(),r['id']) for r in rows)
            similarities.append({'heldout_id':case['id'],'closest_train_id':closest,'character_similarity':round(similarity,4)})
    assert not overlaps
    manifest={'version':'teacher-v3','train':{'examples':len(rows),'response_tokens':sum(r['response_tokens'] for r in rows),
      'total_tokens':sum(r['prompt_tokens']+r['response_tokens'] for r in rows),'max_sequence_tokens':max(r['prompt_tokens']+r['response_tokens'] for r in rows),
      'categories':dict(Counter(r['category'] for r in rows)),'origins':dict(Counter(r['source_kind'] for r in rows)),
      'sha256':digest(ROOT/'train.jsonl')},
      'validation':previous['validation'],'audit':previous['audit'],
      'tokenizer_sha256':digest(TOKENIZER),'weights_updated':False,'paid_api_calls':0,'cloud_jobs_started':0,
      'scope':'Small pilot curriculum, not a complete instruction corpus.',
      'provenance':'Codex-authored and programmatically derived; no independent human review.',
      'heldout_note':'v2 validation is development data with a known baseline. Audit is not run. Shared skills and some output schemas are intentional; lexical separation does not prove semantic independence.'}
    write(ROOT/'manifest.json',manifest)
    write(ROOT/'split-review.json',{'exact_prompt_overlaps':overlaps,'heldout_files_unchanged':True,
        'nearest_prompt_pairs':similarities,'near_match_review_threshold':.85,
        'flagged_pairs':[s for s in similarities if s['character_similarity']>=.85],
        'limitation':'Character similarity is a screening heuristic, not proof of zero contamination.'})
    print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__': main()
