"""Bounded local teacher -> checked data -> train -> evaluate pipeline; no Codex."""
import argparse,datetime,json,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent
HOME=ROOT/'automation-local'
INITIAL=ROOT/'local-run-20261007-190808-8c59a4'
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(path)
def teacher(provider,seed,log):
    with log.open('w',encoding='utf-8') as output:
        result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/f'{provider}-teacher/teacher.ps1'),'-Seed',str(seed)],stdout=output,stderr=subprocess.STDOUT)
    path=ROOT/f'{provider}-teacher/batch-{seed}/report.json'
    report=read(path) if path.exists() else {'status':'failed'}
    return result.returncode==0 and report.get('accepted',0)>0,report
def main():
    p=argparse.ArgumentParser(); p.add_argument('--rounds',type=int,default=1); p.add_argument('--smoke',action='store_true'); p.add_argument('--check',action='store_true'); args=p.parse_args()
    if not 1<=args.rounds<=3: p.error('Escolha de 1 a 3 rodadas')
    HOME.mkdir(exist_ok=True)
    state=HOME/'state.json'
    base=Path(read(state)['base_run']) if state.exists() else INITIAL
    if not base.resolve().is_relative_to(ROOT.resolve()): raise ValueError('Checkpoint fora do projeto')
    for path in [base/'candidate/best-model.pt',base/'evaluation.json',base/'train.jsonl',ROOT/'groq-teacher/teacher.ps1',ROOT/'openrouter-teacher/teacher.ps1']:
        if not path.exists(): raise FileNotFoundError(path)
    if args.check:
        import train_local
        assert train_local.sha(base/'candidate/best-model.pt')==read(base/'evaluation.json')['checkpoint_sha256']
        print('Configuração OK. Base: '+str(base)+'. Nenhuma API ou treino executado.'); return
    lock=HOME/'active.lock'
    try: handle=lock.open('x')
    except FileExistsError: raise SystemExit('Automação já ativa ou interrompida. Consulte automation-local/active.lock antes de reiniciar.')
    session=HOME/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]); session.mkdir()
    handle.write(str(session)); handle.close()
    status={'status':'running','rounds':[],'codex_calls':0,'chat_model_modified':False,'smoke':args.smoke}
    try:
        if (ROOT/'local-training.lock').exists(): raise RuntimeError('Já existe treinamento local ativo; nenhuma API chamada')
        for n in range(args.rounds):
            print(f'Rodada {n+1}/{args.rounds}: pedindo 24 respostas ao professor...',flush=True)
            seed=int(time.time())%2000000000
            while any((ROOT/f'{pr}-teacher/batch-{seed}').exists() for pr in ['groq','openrouter']): seed+=1
            entry={'seed':seed,'teacher_attempts':[]}; status['rounds'].append(entry)
            # At most one Groq request and one free OpenRouter fallback. Never unbounded retry.
            for provider in ['groq','openrouter']:
                ok,report=teacher(provider,seed,session/f'{n}-{provider}.log')
                entry['teacher_attempts'].append({'provider':provider,'report':report})
                write(session/'status.json',status)
                if ok: break
            else: raise RuntimeError('Professores indisponíveis: rodada encerrada sem treino')
            print(f"{report['accepted']} respostas verificadas. Iniciando treino local e avaliação...",flush=True)
            resultfile=session/f'{n}-result.json'
            command=[sys.executable,'-u',str(ROOT/'train_local.py'),'--base-run',str(base),'--result-file',str(resultfile)]
            if args.smoke: command.append('--smoke')
            with (session/f'{n}-training.log').open('w',encoding='utf-8') as output:
                result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError('Treino falhou; consulte o training.log nesta sessão')
            outcome=read(resultfile); entry['training']=outcome
            improved=outcome['summary'].get('improved_on_development',False)
            if improved and not args.smoke:
                base=Path(outcome['run'])
                write(state,{'base_run':str(base),'reason':'More development answers correct with no lost previous correct answers; chat unchanged'})
            entry['used_for_next_round']=improved and not args.smoke
            write(session/'status.json',status)
            summary=outcome['summary']
            print(f"Resultado: {summary['before_correct']}/12 -> {summary.get('after_correct','sem novo checkpoint')}. Modelo de conversa preservado.",flush=True)
            if not improved or args.smoke:
                status['stop_reason']='Sem ganho comprovado nesta avaliação, ou teste rápido'; break
        status['status']='completed'
    except BaseException as error:
        status.update(status='failed_or_interrupted',error=str(error)); raise
    finally:
        write(session/'status.json',status); lock.unlink(missing_ok=True)
        print('Relatório da automação: '+str(session/'status.json'),flush=True)
if __name__=='__main__': main()
