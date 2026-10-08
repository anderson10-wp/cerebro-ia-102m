"""Score both members of each pair; fixed answers cannot pass a pair."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True,type=Path); p.add_argument('--out',required=True,type=Path); args=p.parse_args()
    cases=ROOT/'paired-evaluation/validation.jsonl'
    manifest=json.loads((cases.parent/'manifest.json').read_text())
    assert hashlib.sha256(cases.read_bytes()).hexdigest()==manifest['validation_sha256']
    if args.out.exists(): raise SystemExit('Arquivo de avaliação já existe; escolha outro nome')
    subprocess.run([sys.executable,str(ROOT/'teacher-v2/evaluate.py'),'--checkpoint',str(args.checkpoint),'--cases',str(cases),'--device','directml','--out',str(args.out)],check=True)
    report=json.loads(args.out.read_text(encoding='utf-8'))
    membership={r['id']:r['pair_id'] for r in map(json.loads,cases.read_text(encoding='utf-8').splitlines())}
    pairs={}
    for r in report['results']: pairs.setdefault(membership[r['id']],[]).append(r['correct'])
    assert len(pairs)==16 and all(len(p)==2 for p in pairs.values())
    report.update(correct_pairs=sum(all(p) for p in pairs.values()),total_pairs=16,constant_baseline_correct=16,constant_baseline_correct_pairs=0)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Acertos: {report['correct']}/32; pares inteiros: {report['correct_pairs']}/16.")
if __name__=='__main__': main()
