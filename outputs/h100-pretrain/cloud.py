"""Bounded continuation of our own 102M model. No credentials in the bundle."""
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
os.environ['LIGHTNING_DISABLE_VERSION_CHECK']='1'
from lightning_sdk import Studio, Teamspace, Job, Machine
from lightning_sdk.api.utils import cached_lightning_client
from lightning_sdk.api.cloud_account_api import CloudAccountApi
from lightning_sdk.machine import CloudProvider

ROOT=Path(__file__).resolve().parent
STATE=ROOT/'cloud-state.json'
NAME='cerebro-102m-h100-continuation-nebius'
JOB='cerebro-102m-h100-01'
sys.stdout.reconfigure(encoding='utf-8')

def stamp(): return datetime.now(timezone.utc).isoformat()
def save(s): STATE.write_text(json.dumps(s,indent=2,default=str),encoding='utf-8')

def main():
    mode=sys.argv[1]
    client=cached_lightning_client()
    team=Teamspace('default-project',user=client.auth_service_get_user().username)
    balance=float(client.billing_service_get_project_balance(team.id).balance)
    state=json.loads(STATE.read_text()) if STATE.exists() else {'budget_credits':3,'balance_before':balance,'created_at':stamp()}
    state.update(balance_latest=balance,checked_at=stamp())
    if mode=='inspect':
        api=CloudAccountApi()
        prices=[]
        for cloud in ('lightning-nebius-prod','lightning-nebius-prod','lightning-baremetal'):
            catalog=api.list_cloud_account_accelerators(team.id,cloud,'')
            for a in catalog.accelerator:
                if 'h100' in str(a.family).lower() and a.resources.gpu==1:
                    prices.append({'cloud_account':cloud,**{k:getattr(a,k,None) for k in ('instance_id','slug_multi_cloud','cost','enabled','out_of_capacity','is_tier_restricted','available_in_seconds')}})
        state['h100_catalog']=prices
        save(state)
        print(json.dumps({'balance':balance,'h100_catalog':prices}),flush=True)
    elif mode=='inspect-studios':
        items=[]
        for name in (NAME,'cerebro-102m-h100-staging'):
            studio=Studio(name,teamspace=team,create_ok=False,disable_secrets=True)
            live=studio._studio_api.get_studio_status(studio.id,team.id)
            active=live.in_use or live.requested
            items.append({'name':name,'id':studio.id,'status':str(studio.status),'provider':studio.cloud_account,
                          'price':getattr(active,'price',None),'free':getattr(active,'free',None),
                          'startup_stages':[s.to_dict() for s in (getattr(active,'startup_stages',None) or [])]})
        print(json.dumps(items,default=str),flush=True)
    elif mode=='inspect-endpoint':
        studio=Studio('cerebro-102m-h100-staging',teamspace=team,create_ok=False,disable_secrets=True)
        expected={'staging_server.py','model.py','training_helpers.py','train_h100.py','data-manifest.json','tokenizer.json','staging-report.json'}
        files=[{k:f.get(k) for k in ('path','size')} for f in studio._studio_api.list_files(studio.id,team.id) if Path(f.get('path','')).name in expected]
        print(json.dumps({'status':str(studio.status),'endpoints':[{'id':e.id,'name':e.name,'ports':e.ports,'urls':e.urls,
                          'subdomain':e.lightning_subdomain,'auth_enabled':getattr(e.auth,'enabled',None),
                          'user_api_key':getattr(e.auth,'user_api_key',None)} for e in studio.list_ports()],'files':files}),flush=True)
    elif mode=='prepare':
        bundle=json.loads((ROOT/'bundle-manifest.json').read_text())
        if state.get('job_id'): raise RuntimeError('No duplicate job')
        studio=Studio(NAME,teamspace=team,cloud=CloudProvider.NEBIUS,disable_secrets=True)
        if studio.cloud_account!='lightning-nebius-prod' or str(studio.status)!='Stopped':
            raise RuntimeError('Expected an unstarted Lambda Studio')
        state.update(studio_id=studio.id,provider=studio.cloud_account,status='uploading')
        save(state)
        import hashlib
        for item in bundle['files']:
            path=ROOT/item['local']
            if not path.is_file() or path.stat().st_size!=item['bytes']:
                raise RuntimeError('Incomplete input: '+item['local'])
            with path.open('rb') as f:
                digest=hashlib.file_digest(f,'sha256').hexdigest()
            if digest!=item['sha256']: raise RuntimeError('Changed input: '+item['local'])
            studio.upload_file(str(path),remote_path=item['remote'],progress_bar=False)
            print(json.dumps({'uploaded':item['remote'],'bytes':item['bytes']}),flush=True)
        state.update(status='prepared',studio_status=str(studio.status),bundle_sha256=hashlib.sha256((ROOT/'bundle-manifest.json').read_bytes()).hexdigest())
        save(state)
        print(json.dumps({'status':'prepared','gpu_started':False}),flush=True)
    elif mode=='submit':
        if state.get('status')!='prepared' or state.get('job_id'): raise RuntimeError('Requires prepared inputs and no prior job')
        prices=[m for m in team.list_machines(cloud_account='lightning-nebius-prod') if m.name=='nb-h100-1gpu-16vcpu-200gb']
        if len(prices)!=1 or float(prices[0].cost)>5.70 or balance<3 or state['balance_before']-balance>0.25:
            raise RuntimeError('H100 price or remaining budget changed')
        studio=Studio(NAME,teamspace=team,create_ok=False,disable_secrets=True)
        if str(studio.status)!='Stopped': raise RuntimeError('Source Studio must remain stopped')
        studio.upload_file(str(ROOT/'train_h100.py'),remote_path='train_h100.py',progress_bar=False)
        state.update(budget_credits=27,training_seconds=16200,external_timeout_seconds=16800)
        job=studio.run_job(JOB,prices[0],
            'timeout --signal=TERM --kill-after=30s 16800s python -u /teamspace/studios/this_studio/train_h100.py --seconds 16200',
            interruptible=False,max_run_attempts=1,reuse_snapshot=False)
        state.update(status='submitted',job_id=job.id,job_url=job.link,submitted_at=stamp(),hourly_credits=float(prices[0].cost),max_run_attempts=1)
        save(state)
        print(json.dumps({'status':str(job.status),'url':job.link}),flush=True)
    elif mode=='fetch-eval':
        destination=ROOT/'result'; destination.mkdir(exist_ok=True)
        for name in ('best-model.pt','tokenizer.json','report.json'):
            team.download_file('jobs/'+JOB+'/result/'+name,str(destination/name))
        print(json.dumps({'evaluation_files_downloaded':True,'gpu_started':False}),flush=True)
    elif mode=='fetch-fast':
        import requests,re
        from concurrent.futures import ThreadPoolExecutor
        from lightning_sdk.api.teamspace_api import _authenticate_and_get_auth_headers
        destination=ROOT/'result'; destination.mkdir(exist_ok=True)
        url=team._teamspace_api._client.api_client.configuration.host+'/v1/projects/'+team.id+'/artifacts/blobs/jobs/'+JOB+'/result/best-model.pt'
        headers=_authenticate_and_get_auth_headers()
        with requests.get(url,headers={**headers,'Range':'bytes=0-0'},stream=True,timeout=60) as response:
            value=response.headers.get('Content-Range','')
            if response.status_code!=206 or not re.fullmatch(r'bytes 0-0/\d+',value):
                print(json.dumps({'range_supported':False,'status':response.status_code}),flush=True); return
            size=int(value.split('/')[-1])
        path=destination/'best-model.pt.fast-partial'
        with path.open('wb') as f: f.truncate(size)
        chunk=8*1024**2
        def fetch(start):
            end=min(size,start+chunk)-1
            with requests.get(url,headers={**headers,'Range':f'bytes={start}-{end}'},stream=True,timeout=90) as r:
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}': raise RuntimeError('Invalid download range')
                count=0
                with path.open('r+b') as f:
                    f.seek(start)
                    for block in r.iter_content(256*1024):
                        count+=len(block)
                        if count>end-start+1: raise RuntimeError('Download range too large')
                        f.write(block)
                if count!=end-start+1: raise RuntimeError('Incomplete range')
            return count
        with ThreadPoolExecutor(max_workers=8) as pool:
            total=0
            for amount in pool.map(fetch,range(0,size,chunk)):
                total+=amount; print(json.dumps({'downloaded_bytes':total,'total_bytes':size}),flush=True)
        path.replace(destination/'best-model.pt')
        print(json.dumps({'download_complete':True}),flush=True)
    elif mode=='check':
        if not state.get('job_id'): raise RuntimeError('No submitted job')
        destination=ROOT/'latest-report.json'
        team.download_file('jobs/'+JOB+'/result/report.json',str(destination))
        print(destination.read_text(encoding='utf-8'),flush=True)
    elif mode in ('status','stop','download'):
        if not state.get('job_id'):
            save(state); print(json.dumps(state)); return
        job=Job(JOB,teamspace=team)
        state.update(job_status=str(job.status),job_cost=job.total_cost)
        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(state['submitted_at'])).total_seconds()
        # Stop a stalled allocation early, and bound total spend including startup.
        if mode=='stop' or (mode=='status' and str(job.status) in ('Pending','Running') and
                (job.total_cost>27.0 or state['balance_before']-balance>27.0 or elapsed>18000 or (str(job.status)=='Pending' and elapsed>900))):
            job.stop()
            state.update(job_status=str(job.status),guard_stopped=True)
        if mode=='download':
            remote='jobs/'+JOB+'/result'
            for entry in team.list_files(remote,recursive=True):
                if entry.is_dir: continue
                dest=(ROOT/'result'/entry.path).resolve()
                if (ROOT/'result').resolve() not in dest.parents: raise RuntimeError('Artifact path escaped destination')
                dest.parent.mkdir(parents=True,exist_ok=True)
                team.download_file(remote+'/'+entry.path,str(dest))
                if dest.stat().st_size!=entry.size: raise RuntimeError('Incomplete artifact')
            state['downloaded_at']=stamp()
        save(state)
        print(json.dumps(state,default=str),flush=True)
        if str(job.status)!='Pending': print(job.logs(tail=40,rank=0,timestamps=True),flush=True)

if __name__=='__main__':
    try: main()
    except Exception as error:
        message=str(error)
        if hasattr(error,'status'): message=str(error.status)+' '+str(error.reason)+' '+str(getattr(error,'body',''))
        secret=os.environ.get('LIGHTNING_API_KEY')
        if secret: message=message.replace(secret,'[REDACTED]')
        print(json.dumps({'error_type':type(error).__name__,'error':message}),flush=True)
        sys.exit(1)
