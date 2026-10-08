"""Finish the authorized Kaggle pretraining across bounded private sessions.

Runs on the user's computer; never rewrites model code or accepts web instructions.
Credentials remain in the existing Windows-DPAPI wrapper, outside deliverables.
"""
import argparse, datetime, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

OWNER = 'andersonluan10ad'
CORPUS = OWNER + '/cerebro-v1-102m-corpus'
POLL_SECONDS = 120
RESERVE_HOURS = 3.0
BUDGET_HOURS = 26.0
TARGET = 2_000_000_000

def timestamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()

class Controller:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / 'outputs' / 'cerebro-v1'
        self.work = self.workspace / 'work' / 'training-control'
        self.work.mkdir(parents=True, exist_ok=True)
        self.wrapper = self.workspace / 'work' / 'kaggle.ps1'
        self.state_path = self.root / 'run-state.json'
        self.state = {'started_at': timestamp(), 'status': 'initializing', 'parameters': 102257280,
                      'target_tokens': TARGET, 'reserve_gpu_hours': RESERVE_HOURS,
                      'max_gpu_hours_for_training': BUDGET_HOURS, 'sessions': [], 'controller_pid': os.getpid()}
        self.source_hashes = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                              for name in ['model.py', 'train.py', 'sources.json']}
        if self.state_path.exists():
            saved_state = json.loads(self.state_path.read_text(encoding='utf-8'))
            if saved_state.get('source_sha256') != self.source_hashes:
                raise RuntimeError('Saved training sequence uses different source files')
            if saved_state.get('parameters') != 102257280 or saved_state.get('target_tokens') != TARGET:
                raise RuntimeError('Saved sequence configuration mismatch')
            self.state = saved_state
            self.state['controller_pid'] = os.getpid()
            self.state['resumed_at'] = timestamp()
        self.state['source_sha256'] = self.source_hashes

    def write(self, status, **values):
        self.state.update(status=status, updated_at=timestamp(), **values)
        temp = self.state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.state, indent=2), encoding='utf-8')
        os.replace(temp, self.state_path)
        print(json.dumps({'time': timestamp(), 'status': status, **values}), flush=True)

    def cli(self, *args):
        cmd = ['powershell', '-NoProfile', '-File', str(self.wrapper), *map(str, args)]
        child_env = dict(os.environ)
        # Windows PowerShell must load its own built-in modules rather than the
        # PowerShell 7 module path inherited through the local Python controller.
        child_env['PSModulePath'] = str(Path(os.environ['SystemRoot']) / 'System32' / 'WindowsPowerShell' / 'v1.0' / 'Modules')
        last = None
        for attempt in range(3):
            run = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                 cwd=self.workspace, env=child_env,
                                 timeout=1800 if args[:2] == ('kernels', 'output') else 180)
            if run.returncode == 0:
                return run.stdout
            last = (run.stderr + '\n' + run.stdout)[-4000:]
            # Retry only read-only API operations, never a kernel submission.
            if args[:2] == ('kernels', 'push') or attempt == 2:
                break
            time.sleep(20)
        raise RuntimeError('Kaggle command failed: ' + ' '.join(map(str, args[:3])) + '\n' + last)

    def stopped(self):
        return (self.work / 'stop-after-current-session.flag').exists()

    def status(self, slug):
        result = self.cli('kernels', 'status', slug)
        match = re.search(r'KernelWorkerStatus\.([A-Z_]+)', result)
        if not match:
            raise RuntimeError('Unrecognized kernel status: ' + result)
        return match.group(1)

    def wait(self, slug, max_hours):
        deadline = time.monotonic() + max_hours * 3600
        previous = None
        while time.monotonic() < deadline:
            status = self.status(slug)
            if status != previous:
                print(json.dumps({'time': timestamp(), 'kernel': slug, 'kernel_status': status}), flush=True)
                previous = status
            if status == 'COMPLETE':
                return
            if status in {'ERROR', 'CANCELLED', 'CANCELED'}:
                raise RuntimeError('Kernel failed: ' + slug + ' (' + status + ')')
            if self.stopped():
                # Existing remote run is preserved; only subsequent submissions stop.
                self.write('stop_requested', active_kernel=slug)
                raise RuntimeError('Controller stopped by local stop flag; existing Kaggle run is preserved')
            time.sleep(POLL_SECONDS)
        raise TimeoutError('Kernel wait exceeded expected duration: ' + slug)

    def quota(self):
        data = json.loads(self.cli('quota', '--format', 'json'))
        item = next(x for x in data if x['resource'] == 'GPU')
        return float(item['used'].removesuffix('h')), float(item['remaining'].removesuffix('h')), item['refreshAt']

    def download_report(self, slug, directory):
        self.cli('kernels', 'output', slug, '-p', directory, '--file-pattern', '^brain/report.json$')
        return json.loads((directory / 'brain' / 'report.json').read_text())

    def prepare_session(self, index, previous, hours):
        # Reject accidental local code edits during an active training sequence.
        for name, digest in self.source_hashes.items():
            if hashlib.sha256((self.root / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Training source changed during execution: ' + name)
        directory = self.work / f'session-{index:02d}'
        directory.mkdir(exist_ok=True)
        payload = {name: (self.root / name).read_text(encoding='utf-8') for name in ['model.py', 'train.py']}
        launcher = '''import hashlib, json, os, subprocess, sys
from pathlib import Path
os.environ['NCCL_P2P_DISABLE']='1'
os.environ['NCCL_IB_DISABLE']='1'
root=Path('/kaggle/working/source'); root.mkdir(exist_ok=True)
for name,content in json.loads(PAYLOAD).items(): (root/name).write_text(content)
matches=[]
for path in Path('/kaggle/input').rglob('manifest.json'):
    try:
        meta=json.loads(path.read_text())
        if set(meta.get('counts',{}))=={'pt','en','code'} and 'sources' in meta: matches.append(path.parent)
    except (ValueError,OSError): pass
if len(matches)!=1: raise RuntimeError('Expected exactly one verified corpus mount: '+str(matches))
data=matches[0]
checkpoints=list(Path('/kaggle/input').rglob('last.pt'))
if EXPECT_RESUME and len(checkpoints)!=1: raise RuntimeError('Previous checkpoint missing or ambiguous')
if not EXPECT_RESUME and checkpoints: raise RuntimeError('Unexpected checkpoint in initial run')
cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(root/'train.py'),'--data',str(data),'--hours',str(HOURS),'--target-tokens',str(TARGET)]
if EXPECT_RESUME: cmd.extend(['--resume',str(checkpoints[0])])
subprocess.run(cmd,check=True,cwd=root)
'''.replace('PAYLOAD', repr(json.dumps(payload))).replace('EXPECT_RESUME', repr(bool(previous))).replace('HOURS', repr(hours)).replace('TARGET', repr(TARGET))
        (directory / 'run.py').write_text(launcher, encoding='utf-8')
        slug = f'{OWNER}/cerebro-v1-102m-train-{index:02d}'
        meta = {'id': slug, 'title': f'Cerebro V1 102M Train {index:02d}', 'code_file': 'run.py',
                'language': 'python', 'kernel_type': 'script', 'is_private': True,
                'enable_gpu': True, 'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4',
                'dataset_sources': [], 'competition_sources': [],
                'kernel_sources': [CORPUS] + ([previous] if previous else []), 'model_sources': []}
        (directory / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
        return directory, slug

    def run(self):
        checks = json.loads((self.root / 'gpu-integration.json').read_text())
        if checks.get('status') != 'PASS':
            raise RuntimeError('GPU integration tests have not passed')
        self.write('preparing_corpus', corpus_url='https://www.kaggle.com/code/' + CORPUS)
        self.wait(CORPUS, 7)
        current_used, remaining, current_refresh = self.quota()
        initial_used = self.state.get('initial_gpu_used_hours', current_used)
        refresh = self.state.get('quota_refresh', current_refresh)
        if current_refresh != refresh:
            raise RuntimeError('Quota period changed; refusing to extend the original budget')
        self.state['initial_gpu_used_hours'] = initial_used
        self.state['quota_refresh'] = refresh
        previous, report = None, None
        # Reconcile previously submitted sessions after a computer shutdown.
        # Never resubmit an existing session or reset the original GPU budget.
        for index, session in enumerate(self.state['sessions'], 1):
            slug = session['kernel']
            self.write('recovering_session', active_kernel=slug)
            self.wait(slug, session['max_hours'] + 1)
            report = self.download_report(slug, self.work / f'result-{index:02d}')
            if not report.get('training_from_scratch') or report['parameters'] != 102257280:
                raise RuntimeError('Unexpected recovered training report')
            session['report'] = report
            previous = slug
            self.write('session_recovered', tokens_processed=report['tokens_processed'], last_report=report)
        start_index = len(self.state['sessions']) + 1
        for index in range(start_index, 5):
            if report and report['status'] == 'target_reached':
                break
            used, remaining, current_refresh = self.quota()
            if current_refresh != refresh:
                raise RuntimeError('Quota period changed; refusing to extend the authorized budget automatically')
            consumed = max(0.0, used - initial_used)
            hours = min(8.0, remaining - RESERVE_HOURS - 0.3, BUDGET_HOURS - consumed - 0.3)
            if self.stopped() or hours < 0.25:
                self.write('budget_stop', remaining_gpu_hours=remaining, last_report=report)
                break
            directory, slug = self.prepare_session(index, previous, hours)
            timeout = int((hours + 0.25) * 3600)
            self.write('submitting_training', active_kernel=slug, remaining_gpu_hours=remaining,
                       planned_session_hours=hours, tokens_processed=report['tokens_processed'] if report else 0)
            result = self.cli('kernels', 'push', '-p', directory, '--accelerator', 'NvidiaTeslaT4', '-t', timeout)
            if 'successfully pushed' not in result:
                raise RuntimeError('Submission was not confirmed: ' + result)
            self.state['sessions'].append({'kernel': slug, 'url': 'https://www.kaggle.com/code/' + slug,
                                           'submitted_at': timestamp(), 'max_hours': hours})
            self.write('training', active_kernel=slug, active_url='https://www.kaggle.com/code/' + slug)
            self.wait(slug, hours + 1)
            report_path = self.work / f'result-{index:02d}'
            report = self.download_report(slug, report_path)
            if not report.get('training_from_scratch') or report['parameters'] != 102257280:
                raise RuntimeError('Unexpected training report')
            self.state['sessions'][-1]['report'] = report
            previous = slug
            self.write('session_complete', tokens_processed=report['tokens_processed'], last_report=report)
            if report['status'] == 'target_reached':
                break
        if previous:
            self.write('downloading_weights', final_kernel=previous)
            pattern = r'^brain/(model\.pt|best-model\.pt|tokenizer\.json|config\.json|report\.json|metrics\.jsonl|data-manifest\.json)$'
            self.cli('kernels', 'output', previous, '-p', self.root / 'trained', '--file-pattern', pattern)
            self.write('complete' if report['status'] == 'target_reached' else 'partial_budget_limit',
                       tokens_processed=report['tokens_processed'], final_report=report,
                       checkpoint=str(self.root / 'trained' / 'brain' / 'best-model.pt'))
        else:
            self.write('no_training_budget', remaining_gpu_hours=remaining)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    args = parser.parse_args()
    controller = Controller(args.workspace)
    import msvcrt
    lock = (controller.work / 'controller.lock').open('a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise RuntimeError('A training controller is already active')
    try:
        controller.run()
    except Exception as error:
        controller.write('error', error=str(error))
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()

if __name__ == '__main__':
    main()
