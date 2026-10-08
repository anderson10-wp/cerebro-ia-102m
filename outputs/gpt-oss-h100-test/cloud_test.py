"""Bounded single-H100 comparison; cloud credentials never enter the bundle."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ['LIGHTNING_DISABLE_VERSION_CHECK'] = '1'
from lightning_sdk import Studio, Teamspace, Machine, Job
from lightning_sdk.machine import CloudProvider
from lightning_sdk.api.cloud_account_api import CloudAccountApi
from lightning_sdk.api.utils import cached_lightning_client

ROOT = Path(__file__).resolve().parent
STATE = ROOT/'cloud-state.json'
NAME = 'gpt-oss-h100-comparison'
JOB = 'gpt-oss-120b-vs-20b-h100-01'
sys.stdout.reconfigure(encoding='utf-8')

def save(s):
    STATE.write_text(json.dumps(s, indent=2, default=str), encoding='utf-8')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['inspect','prepare','submit','status','download','stop','billing','complete-transfer','resume','capacity'])
    args = p.parse_args()
    client = cached_lightning_client()
    account = client.auth_service_get_user()
    team = Teamspace('default-project', user=account.username)
    balance = float(client.billing_service_get_project_balance(team.id).balance)
    state = json.loads(STATE.read_text()) if STATE.exists() else {'budget_credits':3,'balance_before':balance}
    state['balance_latest'] = balance
    state['checked_at'] = datetime.now(timezone.utc).isoformat()
    save(state)
    if args.mode=='capacity':
        api=CloudAccountApi()
        fields=('display_name','instance_id','slug_multi_cloud','cost','enabled','out_of_capacity','full','is_tier_restricted','max_available_quota','quota_value','quota_utilization','available_in_seconds')
        items=[]
        for cloud in ('lightning-lambda-prod','lightning-nebius-prod','lightning-baremetal'):
            value=api.list_cloud_account_accelerators(team.id,cloud,'')
            items.extend({'cloud_account':cloud,**{k:getattr(a,k,None) for k in fields}} for a in value.accelerator if 'h100' in str(a.family).lower() or 'h100' in str(a.display_name).lower())
        (ROOT/'capacity.json').write_text(json.dumps(items,indent=2),encoding='utf-8')
        print(json.dumps(items),flush=True)
        return
    if args.mode=='resume':
        studio=Studio(NAME,teamspace=team,create_ok=False,disable_secrets=True)
        if studio.cloud_account!='lightning-lambda-prod' or state['balance_before']-balance>2:
            raise RuntimeError('Expected the transferred Lambda Studio within budget')
        ready_deadline=time.monotonic()+420
        try:
            while str(studio.status)=='Pending' and time.monotonic()<ready_deadline:
                time.sleep(10)
            if str(studio.status)!='Running' or studio.machine!=Machine.H100:
                raise RuntimeError('Single H100 did not become ready')
            state.update({'status':'testing_on_h100','studio_cloud_account':studio.cloud_account,'inference_started_at':datetime.now(timezone.utc).isoformat(),
                          'command_timeout_seconds':600})
            save(state)
            print(json.dumps({'event':'h100_ready','machine':str(studio.machine),'provider':studio.cloud_account}),flush=True)
            print(studio.run('timeout --signal=TERM --kill-after=30s 600s python -u /teamspace/studios/this_studio/compare.py')[-10000:],flush=True)
            state['status']='test_command_finished'
        finally:
            if str(studio.status) in ('Running','Pending'):
                studio.stop()
            state['studio_status']=str(studio.status)
            state['studio_machine']=str(studio.machine)
            state['balance_latest']=float(client.billing_service_get_project_balance(team.id).balance)
            save(state)
            print(json.dumps({'event':'studio_stopped','status':state['studio_status'],'balance':state['balance_latest']}),flush=True)
        return
    if args.mode=='complete-transfer':
        studio=Studio(NAME,teamspace=team,create_ok=False,disable_secrets=True)
        meta=studio._studio.transfer_metadata
        if meta is None or meta.target_cluster_id!='lightning-lambda-prod' or state['balance_before']-balance>2:
            raise RuntimeError('Expected the existing Lambda transfer within budget')
        client.cloud_space_service_complete_running_cloud_space_instance_transfer(team.id,studio.id)
        state['transfer_completion_requested_at']=datetime.now(timezone.utc).isoformat()
        save(state)
        print(json.dumps({'event':'transfer_completion_requested'}),flush=True)
        return
    if args.mode=='billing':
        usage=client.cloud_space_service_get_cloud_space_transfer_estimate(team.id,state['studio_id'],target_cluster_id='lightning-lambda-prod')
        value={'balance':balance,'usage':usage.to_dict()}
        (ROOT/'billing-details.json').write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')
        print(json.dumps(value,default=str),flush=True)
        return
    if args.mode == 'inspect':
        api = CloudAccountApi()
        items = [{'id': c.id, 'provider': str(api._get_cloud_account_provider(c)), 'type': str(c.spec.cluster_type)} for c in api.list_cloud_accounts(team.id)]
        print(json.dumps({'balance':balance, 'cloud_accounts':items, 'h100_prices':[{'name':m.name,'cost':float(m.cost),'provider':str(m.provider)} for m in team.list_machines() if 'h100' in m.name.lower()]}), flush=True)
        if state.get('studio_id'):
            studio = Studio(NAME,teamspace=team,create_ok=False,disable_secrets=True)
            print(json.dumps({'studio_status':str(studio.status),'studio_machine':str(studio.machine),'cloud_account':studio.cloud_account}),flush=True)
            if str(studio.status)=='Running':
                print(studio.run("python -c 'from pathlib import Path; import json; p=Path(\"runtime\"); print(json.dumps({\"runtime_files\":[{\"name\":f.name,\"bytes\":f.stat().st_size} for f in p.glob(\"*\") if f.is_file()]}))'"),flush=True)
        return
    if args.mode == 'prepare':
        if state.get('job_id'):
            raise RuntimeError('An existing job prevents duplicate preparation')
        prices = [m for m in team.list_machines() if m.name == 'gpu_1x_h100_sxm5']
        if len(prices)!=1 or float(prices[0].cost)>4.70 or balance<3:
            raise RuntimeError('H100 price or available budget changed')
        state['hourly_h100_credits'] = float(prices[0].cost)
        studio = Studio(NAME,teamspace=team,cloud=CloudProvider.LIGHTNING_AGGREGATE,disable_secrets=True)
        state['studio_id'] = studio.id
        state['studio_cloud_account'] = studio.cloud_account
        save(state)
        print(json.dumps({'studio_created':studio.id,'cloud_account':studio.cloud_account}),flush=True)
        if str(studio.status)=='Stopped':
            studio.start(Machine.CPU,interruptible=False)
        studio.auto_sleep = True
        if not studio.auto_sleep or studio.auto_sleep_time>600:
            raise RuntimeError('Expected default idle shutdown within ten minutes')
        state['studio_status'] = str(studio.status)
        state['status'] = 'cpu_preparation'
        save(state)
        for name in ('prepare_runtime.py','compare.py','cases.json'):
            studio.upload_file(str(ROOT/name),remote_path=name,progress_bar=False)
        print(studio.run('pwd && df -h .'),flush=True)
        print(studio.run('timeout --signal=TERM --kill-after=20s 1500s python -u prepare_runtime.py')[-5000:],flush=True)
        state['status'] = 'runtime_prepared'
        save(state)
    elif args.mode == 'submit':
        if state.get('job_id') or state.get('run_name') or state.get('status')!='runtime_prepared':
            raise RuntimeError('Requires prepared files and no prior job')
        if state['balance_before']-balance>0.50 or balance<2.5:
            raise RuntimeError('Preparation exceeded its allowance')
        studio = Studio(NAME,teamspace=team,create_ok=False,disable_secrets=True)
        studio.upload_file(str(ROOT/'compare.py'),remote_path='compare.py',progress_bar=False)
        verification=studio.run("python -c 'import hashlib,json; from pathlib import Path; print(json.dumps({n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in (\"compare.py\",\"cases.json\")}))'")
        remote_hashes=json.loads(verification.strip())
        expected_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('compare.py','cases.json')}
        if remote_hashes!=expected_hashes:
            raise RuntimeError('Uploaded comparison code or frozen cases differ')
        state['verified_remote_sha256']=remote_hashes
        print(json.dumps({'event':'comparison_bundle_verified'}),flush=True)
        # Switch the prepared Studio explicitly to Lambda's single H100.
        # The SDK pins Studio jobs to their source cloud; using an AWS source
        # job would not select Lambda's verified 4.68-credit machine.
        prices=[m for m in team.list_machines() if m.name=='gpu_1x_h100_sxm5']
        if len(prices)!=1 or float(prices[0].cost)>4.70:
            raise RuntimeError('H100 price changed before provisioning')
        state.update({'run_name':JOB,'submitted_at':datetime.now(timezone.utc).isoformat(),
                      'status':'switching_to_h100','command_timeout_seconds':900,'max_run_attempts':1})
        save(state)
        try:
            studio.switch_machine(Machine.H100,interruptible=False,cloud_provider=CloudProvider.LAMBDA_LABS)
            if studio.machine!=Machine.H100 or studio.cloud_account!='lightning-lambda-prod':
                raise RuntimeError('Unexpected H100 provider or machine')
            state['status']='testing_on_h100'
            state['studio_cloud_account']=studio.cloud_account
            save(state)
            print(json.dumps({'event':'h100_ready','machine':str(studio.machine),'provider':studio.cloud_account}),flush=True)
            print(studio.run('timeout --signal=TERM --kill-after=30s 900s python -u /teamspace/studios/this_studio/compare.py')[-10000:],flush=True)
            state['status']='test_command_finished'
            save(state)
        finally:
            if str(studio.status) in ('Running','Pending'):
                studio.stop()
            state['studio_status']=str(studio.status)
            state['studio_machine']=str(studio.machine)
            state['balance_latest']=float(client.billing_service_get_project_balance(team.id).balance)
            save(state)
            print(json.dumps({'event':'studio_stopped','status':state['studio_status'],'balance':state['balance_latest']}),flush=True)
    elif args.mode in ('status','download','stop'):
        if state.get('run_name'):
            studio=Studio(NAME,teamspace=team,create_ok=False,disable_secrets=True)
            state.update({'studio_status':str(studio.status),'studio_machine':str(studio.machine),'studio_cloud_account':studio.cloud_account})
            live=studio._studio_api.get_studio_status(studio.id,team.id)
            meta=studio._studio.transfer_metadata
            if meta is not None:
                state['transfer_details']={k:str(getattr(meta,k,None)) for k in ('expected_completion_time','initial_transfer_started_at','top_up_transfer_started_at','source_deletion_started_at','source_cluster_id','target_cluster_id','target_cloud_provider','start_after_transfer')}
            state['studio_sync']={k:str(getattr(studio._studio,k,None)) for k in ('sync_duration','sync_percentage','total_size_bytes','message')}
            if live.requested is not None:
                state['requested_startup']=live.requested.startup_status.to_dict() if live.requested.startup_status else None
                state['requested_details']={k:getattr(live.requested,k,None) for k in ('cluster_id','price','phase','startup_phase','startup_percentage','status_message','bytes_to_sync')}
                state['requested_alerts']=[a.to_dict() for a in (live.requested.alerts or [])]
                state['requested_compute']={k:getattr(live.requested.compute_config,k,None) for k in ('name','count','cluster_override')}
            if live.in_use is not None:
                state['in_use_details']={k:getattr(live.in_use,k,None) for k in ('cluster_id','price','phase','free','status_message','bytes_to_sync')}
                state['active_startup']={k:getattr(live.in_use,k,None) for k in ('startup_phase','startup_percentage','startup_eta_seconds')}
                state['active_compute']={k:getattr(live.in_use.compute_config,k,None) for k in ('name','count','cluster_override')}
                if live.in_use.transfer_metadata is not None:
                    state['transfer_details']={k:str(getattr(live.in_use.transfer_metadata,k,None)) for k in ('expected_completion_time','initial_transfer_started_at','source_cluster_id','target_cluster_id','target_cloud_provider','start_after_transfer')}
            elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(state['submitted_at'])).total_seconds()
            if args.mode=='stop' or (args.mode=='status' and (elapsed>1800 or state['balance_before']-balance>2.60) and str(studio.status) in ('Running','Pending')):
                studio.stop()
                state['stopped_by_budget_guard']=True
            if args.mode=='download':
                remote='studios/'+NAME+'/result'
                entries=team.list_files(remote,recursive=True)
                (ROOT/'result').mkdir(exist_ok=True)
                for entry in entries:
                    if entry.path.endswith(('.json','.log')) and entry.size<10_000_000:
                        dest=(ROOT/'result'/entry.path).resolve()
                        if ROOT.resolve() not in dest.parents:
                            raise ValueError('Artifact path escaped test directory')
                        team.download_file(remote+'/'+entry.path,str(dest))
                        if dest.stat().st_size!=entry.size:
                            raise RuntimeError('Incomplete artifact: '+entry.path)
                state['artifacts_downloaded_at']=datetime.now(timezone.utc).isoformat()
            save(state)
            print(json.dumps(state,default=str),flush=True)
            return
        if not state.get('job_id'):
            print(json.dumps(state),flush=True)
            return
        job = Job(JOB,teamspace=team)
        state.update({'job_status':str(job.status),'job_cost':job.total_cost})
        if args.mode=='stop':
            job.stop()
        if args.mode=='status' and state['job_status'] in ('Running','Pending'):
            elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(state['submitted_at'])).total_seconds()
            if elapsed>2100 or state['balance_before']-balance>2.60:
                job.stop()
                state['stopped_by_budget_guard'] = True
        if state['job_status']=='Completed':
            state['status']='completed'
        save(state)
        if args.mode=='download':
            remote = 'jobs/'+JOB+'/result'
            files = team.list_files(remote,recursive=True)
            (ROOT/'result').mkdir(exist_ok=True)
            for entry in files:
                if entry.path.endswith(('.json','.log')) and entry.size<10_000_000:
                    dest = (ROOT/'result'/entry.path).resolve()
                    if ROOT.resolve() not in dest.parents:
                        raise ValueError('Artifact path escaped test directory')
                    team.download_file(remote+'/'+entry.path,str(dest))
            state['artifacts_downloaded_at']=datetime.now(timezone.utc).isoformat()
            save(state)
        print(json.dumps(state),flush=True)
        if args.mode=='status' and state['job_status']!='Pending':
            print(job.logs(tail=14),flush=True)

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        message = str(error)
        if hasattr(error,'status') and hasattr(error,'reason'):
            message = str(error.status)+' '+str(error.reason)+' '+str(getattr(error,'body',''))
        for key in ('LIGHTNING_API_KEY',):
            secret=os.environ.get(key)
            if secret:
                message=message.replace(secret,'[REDACTED]')
        print(json.dumps({'error_type':type(error).__name__,'error':message}),flush=True)
        raise SystemExit(1)
