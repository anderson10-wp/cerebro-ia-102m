"""Private Lightning pilot, with an OS job timeout and one run attempt."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
os.environ['LIGHTNING_DISABLE_VERSION_CHECK'] = '1'
from lightning_sdk import Studio, Teamspace, Machine, Job
from lightning_sdk.machine import CloudProvider
from lightning_sdk.api.utils import cached_lightning_client

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'lightning-pilot-state.json'
STUDIO = 'cerebro-102m-instruction-pilot'
JOB = 'cerebro-102m-instructions-01'
sys.stdout.reconfigure(encoding='utf-8')

def save(value):
    STATE.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'upload', 'inspect', 'check', 'submit', 'status', 'download', 'stop'])
    args = p.parse_args()
    client = cached_lightning_client()
    account = client.auth_service_get_user()
    team = Teamspace('default-project', user=account.username)
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        'studio': STUDIO, 'job': JOB, 'budget_credits': 2, 'created_at': datetime.now(timezone.utc).isoformat()}
    balance = float(client.billing_service_get_project_balance(team.id).balance)
    state.setdefault('balance_before', balance)
    state['balance_latest'] = balance
    save(state)
    if args.mode == 'prepare':
        prices = [m for m in team.list_machines() if m.name == 'g4dn.xlarge' and str(m.provider) == 'AWS']
        if len(prices) != 1 or float(prices[0].cost) > 0.60 or balance < 2:
            raise RuntimeError('Machine price or available budget changed')
        state['t4_hourly_credits'] = float(prices[0].cost)
        save(state)
        studio = Studio(STUDIO, teamspace=team, cloud=CloudProvider.AWS, disable_secrets=True)
        studio.show_progress = False
        state['studio_id'] = studio.id
        save(state)
        if str(studio.status) == 'Stopped':
            studio.start(Machine.CPU, interruptible=False)
        studio.auto_sleep = True
        if not studio.auto_sleep or studio.auto_sleep_time > 600:
            raise RuntimeError('Expected the default idle shutdown within ten minutes')
        state['auto_sleep'] = studio.auto_sleep
        state['auto_sleep_seconds'] = studio.auto_sleep_time
        state['status'] = 'cpu_preparation'
        save(state)
        print(json.dumps({'status': state['status'], 'auto_sleep_seconds': state['auto_sleep_seconds']}), flush=True)
        result = studio.run("python -c 'import sys,torch; print(sys.version); print(torch.__version__)'")
        print(result, flush=True)
        print(studio.run("timeout --signal=TERM --kill-after=30s 600s python -m pip install 'tokenizers==0.22.2'")[-1500:], flush=True)
        state['status'] = 'cpu_ready'
        save(state)
    elif args.mode == 'upload':
        studio = Studio(STUDIO, teamspace=team, create_ok=False, disable_secrets=True)
        if str(studio.status) == 'Stopped':
            studio.start(Machine.CPU, interruptible=False)
        studio.auto_sleep = True
        if not studio.auto_sleep or studio.auto_sleep_time > 600:
            raise RuntimeError('Expected idle shutdown within ten minutes')
        state['status'] = 'uploading'
        save(state)
        print(json.dumps({'event':'upload_started'}), flush=True)
        studio.upload_file(str(ROOT/'pilot-bundle.zip'), remote_path='pilot-bundle.zip', progress_bar=True)
        state['status'] = 'bundle_uploaded'
        save(state)
    elif args.mode == 'inspect':
        studio = Studio(STUDIO, teamspace=team, create_ok=False, disable_secrets=True)
        state.update({'studio_status': str(studio.status), 'studio_machine': str(studio.machine),
                      'studio_checked_at': datetime.now(timezone.utc).isoformat()})
        save(state)
        print(json.dumps({'status': str(studio.status), 'machine': str(studio.machine)}), flush=True)
        if str(studio.status) == 'Running':
            print(studio.run("python -c 'from pathlib import Path; import json; print(json.dumps({\"cwd\":str(Path.cwd()),\"home\":str(Path.home()),\"files\":[p.name for p in Path.cwd().iterdir()]}))'"), flush=True)
    elif args.mode == 'check':
        studio = Studio(STUDIO, teamspace=team, create_ok=False, disable_secrets=True)
        print(studio.run("python -m zipfile -e /teamspace/studios/this_studio/pilot-bundle.zip /teamspace/studios/this_studio/pilot"), flush=True)
        print(studio.run("timeout --signal=TERM --kill-after=30s 180s python /teamspace/studios/this_studio/pilot/outputs/agent-pilot/pilot_train.py --check"), flush=True)
        state['status'] = 'uploaded_and_checked'
        save(state)
    elif args.mode == 'submit':
        if state.get('status') != 'uploaded_and_checked':
            raise RuntimeError('Prepare and check the bundle before submission')
        if state['balance_before'] - balance > 0.5:
            raise RuntimeError('Preparation spent more than the reserved allowance')
        prices = [m for m in team.list_machines() if m.name == 'g4dn.xlarge' and str(m.provider) == 'AWS']
        if len(prices) != 1 or float(prices[0].cost) > 0.60 or balance < 1.5:
            raise RuntimeError('GPU price or remaining budget changed before launch')
        studio = Studio(STUDIO, teamspace=team, create_ok=False, disable_secrets=True)
        # T4 max_runtime is not an enforceable runtime limit. GNU timeout runs
        # inside the job, and a single attempt disables backend retry spending.
        job = studio.run_job(JOB, Machine.T4,
            'timeout --signal=TERM --kill-after=30s 1800s python -u /teamspace/studios/this_studio/pilot/outputs/agent-pilot/pilot_train.py',
            interruptible=False, max_run_attempts=1, reuse_snapshot=False)
        state.update({'status': 'submitted', 'job_id': job.id, 'job_url': job.link,
                      'submitted_at': datetime.now(timezone.utc).isoformat(),
                      'command_timeout_seconds': 1800, 'max_run_attempts': 1})
        save(state)
        if str(studio.status) == 'Running':
            studio.stop()
        print(json.dumps({'status': str(job.status), 'url': job.link}), flush=True)
    elif args.mode in ('status', 'download', 'stop'):
        if 'job_id' not in state:
            print(json.dumps(state), flush=True)
            return
        job = Job(JOB, teamspace=team)
        state.update({'job_status': str(job.status), 'job_cost': job.total_cost,
                      'balance_latest': balance, 'checked_at': datetime.now(timezone.utc).isoformat()})
        if args.mode == 'status' and state['job_status'] in ('Running', 'Pending'):
            elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(state['submitted_at'])).total_seconds()
            if elapsed > 2400 or state['balance_before']-balance > 1.5:
                job.stop()
                state['stopped_by_local_budget_guard'] = True
        if state['job_status'] == 'Completed':
            state['status'] = 'completed'
        save(state)
        if args.mode == 'stop':
            job.stop()
        elif args.mode == 'download':
            remote = 'jobs/'+JOB+'/pilot/outputs/agent-pilot/result'
            entries = team.list_files(remote, recursive=True)
            print(json.dumps({'artifact_entries': [dict(path=e.path, size=e.size) for e in entries]}), flush=True)
            target = ROOT/'lightning-result'
            target.mkdir(exist_ok=True)
            for name in ('report.json', 'before.json', 'after.json', 'best-model.pt'):
                print(json.dumps({'event':'downloading_artifact', 'name':name}), flush=True)
                team.download_file(remote+'/'+name, str(target/name))
                entry = next(e for e in entries if e.path == name)
                if (target/name).stat().st_size != entry.size:
                    raise RuntimeError('Artifact size differs from remote listing: '+name)
            state['artifacts_downloaded_at'] = datetime.now(timezone.utc).isoformat()
            state['artifact_sizes'] = {e.path:e.size for e in entries}
            save(state)
        print(json.dumps(state, default=str), flush=True)
        if args.mode == 'status' and state['job_status'] != 'Pending':
            print(job.logs(tail=12), flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        message = str(error)
        if error.__cause__ is not None:
            message += ' | caused by: '+str(error.__cause__)
        for key in ('LIGHTNING_API_KEY', 'GPT_OSS_LOCAL_KEY'):
            secret = os.environ.get(key)
            if secret:
                message = message.replace(secret, '[REDACTED]')
        print(json.dumps({'error_type': type(error).__name__, 'error': message}, ensure_ascii=True), flush=True)
        raise SystemExit(1)
