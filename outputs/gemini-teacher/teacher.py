"""One bounded Gemini call per run. Checked labels only; never launches training."""
import argparse,datetime,json,os,random,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def tasks(seed):
    rng=random.Random(seed); rows=[]
    def add(category,prompt,expected,scorer='exact'):
        rows.append(dict(id=f'gemini-{seed}-{len(rows):03}',category=category,prompt=f'Pergunta: {prompt}\nResposta:',response=expected,scorer=scorer,split='train'))
    words=['janela','sapato','nuvem','caneta','copo','tigre','pincel','vaso','espelho','violão','cortina','cavalo']
    rng.shuffle(words)
    for i in range(4):
        a,b=rng.randint(20,80),rng.randint(2,15)
        add('arithmetic',f'Escreva apenas o resultado de {a} + {b}.',str(a+b))
        phrase=rng.sample(words,3); idx=rng.randrange(3)
        add('word_selection',f'Texto: {" ".join(phrase)}. Escreva só a palavra número {idx+1}.',phrase[idx])
        add('instruction',f'Copie exatamente uma vez, sem comentários: {words[i]}',words[i])
        place=['na mochila','sobre o banco','no armário','perto da janela'][i]
        add('context',f'Registro {seed}: objeto {words[i]}; local {place}. Responda apenas o local.',place)
        key=['pronto','ocupado','visivel','aberto'][i]; value=bool(i%2)
        add('json',f'Devolva somente JSON com a chave {key} e o booleano {"verdadeiro" if value else "falso"}.',json.dumps({key:value},separators=(',',':')),'json')
        add('grounding',f'Ficha {seed}-{i}: nome {words[i]}; telefone não cadastrado. Informe só o telefone; se ausente, responda "não informado".','não informado')
    return rows
def correct(answer,row):
    if not isinstance(answer,str): return False
    if row['scorer']=='exact': return answer.strip()==row['response']
    try: return json.dumps(json.loads(answer),sort_keys=True,allow_nan=False)==json.dumps(json.loads(row['response']),sort_keys=True)
    except (ValueError,TypeError): return False
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--seed',type=int,default=106); parser.add_argument('--check',action='store_true'); parser.add_argument('--provider',choices=['gemini','groq','openrouter'],default='gemini'); args=parser.parse_args()
    rows=tasks(args.seed)
    if args.provider!='gemini':
        for row in rows: row['id']=row['id'].replace('gemini-',args.provider+'-',1)
    # Holdouts stay local, never sent to Gemini.
    held=set()
    for split in ['validation','audit']:
        held.update(json.loads(line)['prompt'].strip().casefold() for line in (ROOT.parent/f'teacher-v3/{split}.jsonl').read_text(encoding='utf-8').splitlines())
    assert not held & {r['prompt'].strip().casefold() for r in rows}
    assert all(correct(r['response'],r) for r in rows)
    assert not correct('{"a":0}',{'scorer':'json','response':'{"a":false}'})
    if args.check: print('24 cases checked; no API call.'); return
    output_root=ROOT if args.provider=='gemini' else ROOT.parent/(args.provider+'-teacher')
    output_root.mkdir(exist_ok=True)
    out=output_root/f'batch-{args.seed}'
    out.mkdir(exist_ok=False) # Never silently spend again on an existing batch.
    model='gemini-3.8-flash' if args.provider=='gemini' else ('openai/gpt-oss-120b' if args.provider=='groq' else 'openrouter/free')
    prompt='Você é professor de um modelo pequeno em português. Responda cada exercício de forma estritamente correta e curta. Não explique seu raciocínio. Retorne somente JSON no formato {"answers":[{"id":"...","answer":"resposta textual"}]}. Exercícios:\n'+json.dumps([{'id':r['id'],'question':r['prompt']} for r in rows],ensure_ascii=False)
    body={'contents':[{'role':'user','parts':[{'text':prompt}]}],'generationConfig':{'temperature':0,'maxOutputTokens':4096,'responseMimeType':'application/json'}}
    report={'model':model,'requested':len(rows),'max_calls':1,'max_output_tokens':4096,'training_started':False,'billing_tier':'unknown','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    report['provider']=args.provider
    key=os.environ['GEMINI_API_KEY' if args.provider=='gemini' else ('GROQ_API_KEY' if args.provider=='groq' else 'OPENROUTER_API_KEY')]
    if args.provider=='gemini':
        url=f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
        headers={'Content-Type':'application/json','x-goog-api-key':key}
    else:
        url='https://api.groq.com/openai/v1/chat/completions' if args.provider=='groq' else 'https://openrouter.ai/api/v1/chat/completions'
        headers={'Content-Type':'application/json','Authorization':'Bearer '+key,'User-Agent':'LocalTeacher/1.0'}
        body={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_completion_tokens':4096,'response_format':{'type':'json_object'}}
    request=urllib.request.Request(url,data=json.dumps(body).encode(),headers=headers)
    try:
        with urllib.request.urlopen(request,timeout=60) as response: result=json.load(response)
        if args.provider=='gemini':
            report['usage']=result.get('usageMetadata',{})
            candidate=result['candidates'][0]
            if candidate.get('finishReason')!='STOP': raise ValueError('Incomplete generation')
            answer=json.loads(''.join(p.get('text','') for p in candidate['content']['parts'] if not p.get('thought')))
        else:
            report['usage']=result.get('usage',{})
            report['actual_model']=result.get('model',model)
            candidate=result['choices'][0]
            if candidate.get('finish_reason')!='stop': raise ValueError('Incomplete generation')
            answer=json.loads(candidate['message']['content'])
        answers=answer['answers']; ids=[a['id'] for a in answers]
        if len(ids)!=len(set(ids)) or set(ids)!={r['id'] for r in rows}: raise ValueError('Missing, duplicate or unexpected IDs')
        mapping={a['id']:a['answer'] for a in answers}
        accepted=[]; rejected=[]
        for row in rows:
            output=mapping[row['id']]
            if correct(output,row):
                accepted.append(dict(row,response=output.strip(),provenance=report.get('actual_model',model),teacher_seed=args.seed,checked=True,validator='deterministic reference; exact or typed JSON'))
            else: rejected.append(dict(row,teacher_output=output))
        (out/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted),encoding='utf-8')
        (out/'rejected.json').write_text(json.dumps(rejected,ensure_ascii=False,indent=2),encoding='utf-8')
        report.update(status='completed',accepted=len(accepted),rejected=len(rejected))
    except urllib.error.HTTPError as error:
        report.update(status='api_error',http_status=error.code)
        try:
            detail=json.loads(error.read()).get('error',{})
            report['message']=str(detail.get('message','')).replace(key,'[REDACTED]')
        except Exception: pass
    except Exception as error: report.update(status='failed',error_type=type(error).__name__)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    if report['status']!='completed': raise SystemExit(1)
if __name__=='__main__': main()

