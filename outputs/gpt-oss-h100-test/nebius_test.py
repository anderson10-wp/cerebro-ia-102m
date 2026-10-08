"""One bounded H100 attempt without migrating the previous Studio's disk."""
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
os.environ['LIGHTNING_DISABLE_VERSION_CHECK']='1'
from lightning_sdk import Studio, Teamspace, Machine
from lightning_sdk.api.utils import cached_lightning_client
from lightning_sdk.machine import CloudProvider

ROOT=Path(__file__).resolve().parent
STATE=ROOT/'nebius-state.json'
NAME='gpt-oss-h100-direct-test'
sys.stdout.reconfigure(encoding='utf-8')

def save(s):
    STATE.write_text(json.dumps(s,indent=2,default=str),encoding='utf-8')

def main():
    mode=sys.argv[1]
    client=cached_lightning_client()
    team=Teamspace('default-project',user=client.auth_service_get_user().username)
    balance=float(client.billing_service_get_project_balance(team.id).balance)
    state=json.loads(STATE.read_text()) if STATE.exists() else {'balance_before':balance,'original_balance':29.78,'total_budget':3}
    state['balance_latest']=balance
    state['checked_at']=datetime.now(timezone.utc).isoformat()
    if mode=='cleanup':
        deleted=[]
        # Only the two disposable Studios created for this test; preserve
        # the original model, instruction pilot and all local scripts.
        targets=[(NAME,'01m46kzkwzw6q29v1jnrp27j8f','nebius-state.json'),
                 ('gpt-oss-h100-comparison','01m46hyys1s47d6d8twq5cb6gh','cloud-state.json')]
        for name,expected_id,state_file in targets:
            studio=Studio(name,teamspace=team,create_ok=False,disable_secrets=True)
            recorded=json.loads((ROOT/state_file).read_text())
            if studio.id!=expected_id or recorded['studio_id']!=expected_id or str(studio.status)!='Stopped':
                raise RuntimeError('Temporary Studio identity or stopped state changed')
            studio.delete()
            recorded.update(status='cancelled_by_user',studio_status='Deleted',cancelled_at=datetime.now(timezone.utc).isoformat())
            (ROOT/state_file).write_text(json.dumps(recorded,indent=2,default=str),encoding='utf-8')
            deleted.append({'name':name,'id':expected_id})
        result={'status':'cancelled_by_user','deleted_temporary_studios':deleted,'gpu_tests_completed':0,
                'balance_latest':float(client.billing_service_get_project_balance(team.id).balance),
                'checked_at':datetime.now(timezone.utc).isoformat()}
        (ROOT/'cancellation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)
    elif mode=='run':
        if state.get('submitted_at'):
            raise RuntimeError('No duplicate H100 attempts')
        prices=[m for m in team.list_machines() if m.name=='nb-h100-1gpu-16vcpu-200gb']
        if len(prices)!=1 or float(prices[0].cost)>5.70 or 29.78-balance>1.4:
            raise RuntimeError('H100 price or remaining budget changed')
        previous=Studio('gpt-oss-h100-comparison',teamspace=team,create_ok=False,disable_secrets=True)
        if str(previous.status)!='Stopped':
            raise RuntimeError('Previous H100 attempt must be stopped')
        studio=Studio(NAME,teamspace=team,cloud=CloudProvider.NEBIUS,disable_secrets=True)
        if studio.cloud_account!='lightning-nebius-prod':
            raise RuntimeError('Unexpected provider; do not start GPU')
        state.update(studio_id=studio.id,provider=studio.cloud_account,hourly_credits=float(prices[0].cost),
                     submitted_at=datetime.now(timezone.utc).isoformat(),status='starting')
        save(state)
        try:
            studio.start(Machine.H100,interruptible=False)
            state.update(status='running',started_at=datetime.now(timezone.utc).isoformat())
            save(state)
            print(json.dumps({'event':'h100_ready','studio':NAME,'provider':studio.cloud_account}),flush=True)
            if not studio.auto_sleep or studio.auto_sleep_time>600:
                raise RuntimeError('Expected default idle shutdown within ten minutes')
            for name in ('prepare_runtime.py','compare.py','cases.json'):
                studio.upload_file(str(ROOT/name),remote_path=name,progress_bar=False)
            # Download and inference share one 15-minute wall clock limit.
            command="timeout --signal=TERM --kill-after=20s 900s bash -c 'python -u prepare_runtime.py && python -u compare.py'"
            print(studio.run(command)[-18000:],flush=True)
            state['status']='command_finished'
        finally:
            if str(studio.status) in ('Running','Pending'):
                studio.stop()
            state['studio_status']=str(studio.status)
            state['balance_latest']=float(client.billing_service_get_project_balance(team.id).balance)
            save(state)
            print(json.dumps({'event':'stopped','status':state['studio_status'],'balance':state['balance_latest']}),flush=True)
    elif mode in ('status','stop','download'):
        studio=Studio(NAME,teamspace=team,create_ok=False,disable_secrets=True)
        state['studio_status']=str(studio.status)
        live=studio._studio_api.get_studio_status(studio.id,team.id)
        active=live.in_use or live.requested
        if active:
            state['active']={k:getattr(active,k,None) for k in ('phase','price','startup_phase','startup_percentage','startup_eta_seconds','status_message','instance_id','instance_region','creation_timestamp','start_timestamp','capacity_reservation_id')}
            state['startup_status']=active.startup_status.to_dict() if active.startup_status else None
            state['startup_stages']=[s.to_dict() for s in (active.startup_stages or [])]
            state['compute']=active.compute_config.to_dict() if active.compute_config else None
            state['alerts']=[a.to_dict() for a in (active.alerts or [])]
        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(state['submitted_at'])).total_seconds()
        # Count Pending time conservatively; a billing balance can update late.
        if mode=='stop' or (mode=='status' and (29.78-balance>2.90 or elapsed>960)):
            if str(studio.status) in ('Running','Pending'):
                studio.stop()
                state['studio_status']=str(studio.status)
                state['guard_stopped']=True
        if mode=='download':
            base='studios/'+NAME+'/result'
            for entry in team.list_files(base,recursive=True):
                if not entry.is_dir and entry.path.endswith(('.json','.log')) and entry.size<10000000:
                    dest=(ROOT/'result'/entry.path).resolve()
                    if (ROOT/'result').resolve() not in dest.parents:
                        raise RuntimeError('Unexpected artifact path')
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    team.download_file(base+'/'+entry.path,str(dest))
                    if dest.stat().st_size!=entry.size:
                        raise RuntimeError('Incomplete artifact')
            state['downloaded_at']=datetime.now(timezone.utc).isoformat()
        save(state)
        print(json.dumps(state,default=str),flush=True)

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        message=str(error)
        if hasattr(error,'status'):
            message=str(error.status)+' '+str(error.reason)+' '+str(getattr(error,'body',''))
        secret=os.environ.get('LIGHTNING_API_KEY')
        if secret:
            message=message.replace(secret,'[REDACTED]')
        print(json.dumps({'error_type':type(error).__name__,'error':message}),flush=True)
        sys.exit(1)
