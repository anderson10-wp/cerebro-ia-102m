"""Offline single-round training and evaluation. No API calls or automatic promotion."""
import argparse,datetime,hashlib,json,shutil,subprocess,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def run(command,log):
    with log.open('w',encoding='utf-8') as out:
        result=subprocess.run([sys.executable,'-u',*map(str,command)],stdout=out,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'Etapa falhou. Consulte {log}')
def canonical(row):
    if row.get('scorer')!='json': return row['response']
    try: return json.dumps(json.loads(row['response']),ensure_ascii=False,separators=(',',':'))
    except json.JSONDecodeError: raise ValueError('JSON invalido: '+row['prompt'])
def prepare(folder,seconds,base_run=None):
    from tokenizers import Tokenizer
    base=ROOT/'teacher-v5/candidate/best-model.pt'
    baseline=ROOT/'teacher-v5/candidate-validation.json'
    if base_run is not None:
        base_run=base_run.resolve()
        base=base_run/'candidate/best-model.pt'
        baseline=base_run/'evaluation.json'
    assert sha(base)==read(baseline)['checkpoint_sha256'], 'Checkpoint diferente do avaliado'
    tokenizer=ROOT/'h100-pretrain/result/tokenizer.json'
    tok=Tokenizer.from_file(str(tokenizer))
    replay=(base_run if base_run is not None else ROOT/'teacher-v5')/'train.jsonl'
    rows=[json.loads(line) for line in replay.read_text(encoding='utf-8-sig').splitlines()]
    for row in rows: row['response']=canonical(row)
    held=set()
    for split in ['validation','audit']:
        source=ROOT/f'teacher-v3/{split}.jsonl'
        assert sha(source)==read(ROOT/'teacher-v3/manifest.json')[split]['sha256']
        held.update(json.loads(line)['prompt'].strip().casefold() for line in source.read_text(encoding='utf-8').splitlines())
        shutil.copyfile(source,folder/f'{split}.jsonl')
    seen={r['prompt'].strip().casefold():r['response'] for r in rows}
    assert not set(seen)&held, 'Replay sobreposto à avaliação'
    added=0
    for provider in ['groq','openrouter','gemini','programmatic','curriculum']:
        for batch in sorted((ROOT/f'{provider}-teacher').glob('batch-*')):
            if not (batch/'report.json').exists() or read(batch/'report.json').get('status')!='completed': continue
            for line in (batch/'train.jsonl').read_text(encoding='utf-8').splitlines():
                row=json.loads(line); key=row['prompt'].strip().casefold()
                if row.get('checked') is not True: raise ValueError('Exemplo externo sem validação')
                if key in held: raise ValueError('Sobreposição com avaliação reservada')
                if key in seen:
                    if seen[key]!=canonical(row): raise ValueError('Respostas conflitantes para a mesma pergunta')
                    continue
                row['response']=canonical(row)
                assert len(tok.encode(row['prompt']+row['response']+'<|eos|>').ids)<=1024
                seen[key]=row['response']; rows.append(row); added+=1
    if len(rows)>2000: raise ValueError('Mais de 2000 exemplos: ajuste explícito necessário')
    (folder/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    manifest={s:{'sha256':sha(folder/f'{s}.jsonl')} for s in ['train','validation','audit']}
    manifest.update(tokenizer_sha256=sha(tokenizer),examples=len(rows),teacher_examples_added=added)
    write(folder/'manifest.json',manifest)
    recipe=read(ROOT/'teacher-v5/pilot-config.json')
    recipe.update(base_checkpoint=str(base),base_checkpoint_sha256=sha(base),tokenizer=str(tokenizer),train_sha256=manifest['train']['sha256'],max_training_seconds=seconds,retention_policy='Full v5 replay plus unique validated teacher examples; start from fixed v5 candidate')
    write(folder/'pilot-config.json',recipe)
    shutil.copyfile(ROOT/'teacher-v5/pilot.py',folder/'pilot.py')
    print(f'{len(rows)} exemplos; {added} dos professores. Pasta: {folder}',flush=True)
    return baseline
def main():
    p=argparse.ArgumentParser(); p.add_argument('--check',action='store_true'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--base-run',type=Path); p.add_argument('--result-file',type=Path); p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--device',choices=['cpu','directml'],default='directml'); args=p.parse_args()
    if not 2<=args.seconds<=600: p.error('Limite de treino: 2 a 600 segundos')
    if args.device=='directml':
        import torch_directml
        names=[torch_directml.device_name(i).rstrip('\x00') for i in range(torch_directml.device_count())]
        if sum('RX 580' in name.upper() for name in names)!=1: raise RuntimeError('RX 580 não encontrada; treino cancelado sem fallback para CPU')
    lock=ROOT/'local-training.lock'
    try: handle=lock.open('x')
    except FileExistsError: raise SystemExit('Outra rodada está ativa ou foi interrompida. Verifique antes de remover outputs/local-training.lock.')
    folder=ROOT/('local-run-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    folder.mkdir()
    try:
        handle.write(str(folder)); handle.close()
        baseline=prepare(folder,2 if args.smoke else args.seconds,args.base_run)
        print('Verificando dados e gradientes...',flush=True)
        run([folder/'pilot.py','--device',args.device],folder/'check.log')
        if args.check:
            print('Verificação concluída. Nenhum treinamento ou chamada de API.'); return
        print(f'Treinando em {args.device}: até 2 épocas ou {2 if args.smoke else args.seconds} segundos. Aguarde...',flush=True)
        run([folder/'pilot.py','--train','--device',args.device],folder/'training.log')
        candidate=folder/'candidate/best-model.pt'
        before=read(baseline)
        summary={'status':'completed','before_correct':before['correct'],'total':before['evaluated'],'promoted':False,'api_calls':0,'smoke_test':args.smoke,'training':read(folder/'candidate/report.json')}
        if candidate.exists():
            print('Avaliando respostas...',flush=True)
            run([ROOT/'teacher-v2/evaluate.py','--checkpoint',candidate,'--out',folder/'evaluation.json'],folder/'evaluation.log')
            after=read(folder/'evaluation.json'); old={r['id']:r for r in before['results']}
            regressions=[r['id'] for r in after['results'] if old[r['id']]['correct'] and not r['correct']]
            summary.update(after_correct=after['correct'],regressions=regressions,improved_on_development=after['correct']>before['correct'] and not regressions)
            result=f"Resultado: {before['correct']}/{before['evaluated']} antes; {after['correct']}/{after['evaluated']} agora. Regressões: {len(regressions)}."
        else:
            result='A loss não melhorou: nenhum novo checkpoint selecionado.'
        write(folder/'summary.json',summary)
        if args.result_file: write(args.result_file,{'run':str(folder),'summary':summary})
        (folder/'resultado.txt').write_text(result+'\nModelo de conversa preservado. Avaliação de desenvolvimento pequena; não comprova melhoria geral.\n',encoding='utf-8')
        print(result+'\nRelatório: '+str(folder/'resultado.txt'),flush=True)
    except BaseException as error:
        write(folder/'failure.json',{'status':'interrupted_or_failed','error':str(error),'promoted':False})
        raise
    finally:
        handle.close(); lock.unlink(missing_ok=True)
if __name__=='__main__': main()
