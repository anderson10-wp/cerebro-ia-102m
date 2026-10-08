"""Same backend, hardware and prompts; no generated tools or code are executed."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'result'
RUNTIME=ROOT/'runtime'
START=time.monotonic()

def write(value):
    tmp=OUT/'report.json.tmp'
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(OUT/'report.json')

def request(path,key,body=None):
    req=urllib.request.Request('http://127.0.0.1:8081'+path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
    with urllib.request.urlopen(req,timeout=150) as response:
        return json.load(response)

def main():
    OUT.mkdir(exist_ok=False)
    manifest=json.loads((RUNTIME/'manifest.json').read_text())
    cases=json.loads((ROOT/'cases.json').read_text(encoding='utf-8'))
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    if 'H100' not in gpu or len(gpu.splitlines())!=1:
        raise RuntimeError('Expected exactly one H100; found '+gpu)
    report={'status':'starting','started_at':datetime.now(timezone.utc).isoformat(),'gpu':gpu,
            'runtime':manifest,'settings':{'temperature':0,'reasoning_effort':'medium','max_tokens':2048,'context':8192,'parallel':1},
            'case_count_per_model':len(cases),'results':[],'tools_executed':0,'generated_code_executed':0,
            'scope':'Small controlled comparison; not a broad benchmark. Model sizes use the same pinned backend and hardware.'}
    write(report)
    llama=next((RUNTIME/'llama').rglob('llama-server'))
    env=dict(os.environ)
    env['LD_LIBRARY_PATH']=':'.join(sorted({str(p.parent) for p in (RUNTIME/'llama').rglob('*.so*')}))+':'+env.get('LD_LIBRARY_PATH','')
    key=secrets.token_urlsafe(32)
    keypath=Path('/tmp/h100-comparison.key')
    keypath.write_text(key)
    keypath.chmod(0o600)
    server=None
    try:
        # Test the requested 120B first so a time limit preserves its results.
        for model in sorted(manifest['models'],key=lambda m: m['model']!='gpt-oss-120b'):
            if time.monotonic()-START>720:
                raise TimeoutError('Comparison wall-time budget exhausted')
            model_id=model['model']
            logfile=(OUT/(model_id+'.log')).open('w')
            report['status']='loading_'+model_id
            write(report)
            server=subprocess.Popen([str(llama),'-m',str(RUNTIME/model['file']),
                '--host','127.0.0.1','--port','8081','--alias',model_id,
                '-ngl','99','-c','8192','-np','1','-b','256','-ub','128','--jinja',
                '--api-key-file',str(keypath)],env=env,stdout=logfile,stderr=subprocess.STDOUT)
            ready=False
            load_started=time.monotonic()
            while time.monotonic()-load_started<240:
                if server.poll() is not None:
                    raise RuntimeError('Server failed; inspect '+model_id+'.log')
                try:
                    if request('/health',key).get('status')=='ok':
                        ready=True
                        break
                except (urllib.error.URLError,TimeoutError):
                    pass
                time.sleep(2)
            if not ready:
                raise TimeoutError('Loading timed out')
            print(json.dumps({'event':'model_ready','model':model_id,'load_seconds':round(time.monotonic()-load_started,2)}),flush=True)
            report.setdefault('gpu_loaded',{})[model_id]=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
            # Identical unscored warm-up; do not count initial kernel/load effects.
            request('/v1/chat/completions',key,{'model':model_id,'messages':[{'role':'user','content':'Responda somente: pronto.'}],
                    **{k:v for k,v in report['settings'].items() if k in ('temperature','reasoning_effort','max_tokens')}})
            for case in cases:
                if time.monotonic()-START>750:
                    raise TimeoutError('Comparison wall-time budget exhausted')
                settings={k:v for k,v in report['settings'].items() if k in ('temperature','reasoning_effort','max_tokens')}
                body={'model':model_id,'messages':case['messages'],**settings}
                if case.get('tools'):
                    body.update({'tools':case['tools'],'tool_choice':'auto'})
                started=time.monotonic()
                data=request('/v1/chat/completions',key,body)
                message=data['choices'][0]['message']
                result={'id':case['id'],'kind':case['kind'],'model':model_id,
                        'answer':message.get('content'),'tool_calls':message.get('tool_calls'),
                        'finish_reason':data['choices'][0].get('finish_reason'),'usage':data.get('usage'),
                        'seconds':round(time.monotonic()-started,3)}
                # Raw internal reasoning is not exported in the comparison report.
                report['results'].append(result)
                report['status']='testing_'+model_id
                write(report)
                print(json.dumps({'event':'case_complete','model':model_id,'id':case['id'],'seconds':result['seconds'],'finish_reason':result['finish_reason']}),flush=True)
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            server=None
            logfile.close()
        report['status']='complete'
        report['elapsed_seconds']=round(time.monotonic()-START,2)
        write(report)
        print(json.dumps({'event':'complete','results':len(report['results']),'seconds':report['elapsed_seconds']}),flush=True)
    except Exception as error:
        report['status']='failed'
        report['error_type']=type(error).__name__
        report['error']=str(error)
        write(report)
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        keypath.unlink(missing_ok=True)

if __name__=='__main__':
    main()
