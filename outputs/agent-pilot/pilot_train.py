"""One bounded GPU experiment with before/after evaluation and no tool execution."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import torch
from tokenizers import Tokenizer
from sft import load_examples, batch, validation_loss
from routing import SYSTEM, score

ROOT = Path(__file__).resolve().parent
BRAIN = ROOT.parent/'cerebro-v1'
sys.path.insert(0, str(BRAIN))
from model import Brain, Config

REGRESSION_TEXTS = [
    'O Brasil é um país da América do Sul. Brasília é a capital do Brasil.',
    'Uma função recebe argumentos e pode retornar um resultado. Testes ajudam a encontrar erros no código.',
    'Para medir uma melhoria, é necessário comparar os resultados usando os mesmos critérios de avaliação.',
    'Quando uma informação não está disponível, é importante informar essa limitação e pedir os dados necessários.',
    'A computer stores data in memory. A program is a sequence of instructions that performs a task.',
    'Training and testing use separate examples. A correct answer should be checked against independent evidence.',
    'def soma(a, b):\n    return a + b\n\nassert soma(2, 3) == 5\n',
    'def media(valores):\n    if not valores:\n        raise ValueError("lista vazia")\n    return sum(valores) / len(valores)\n']

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

@torch.no_grad()
def regression(net, tok, device):
    net.eval()
    total, count = 0.0, 0
    for text in REGRESSION_TEXTS:
        ids = torch.tensor([tok.encode(text).ids], device=device)
        loss = float(net(ids[:, :-1], ids[:, 1:]))
        total += loss*(ids.shape[1]-1)
        count += ids.shape[1]-1
    return {'weighted_nll': total/count, 'tokens': count, 'documents': len(REGRESSION_TEXTS),
            'note': 'Small fixed language probe; does not establish broad knowledge or reasoning quality.'}

@torch.no_grad()
def evaluate(net, tok, cases, device, label, out):
    net.eval()
    results = []
    eos = tok.token_to_id('<|eos|>')
    for case in cases:
        prompt = '<|system|>'+SYSTEM+'\nDisponíveis: '+', '.join(case['available'])+'<|user|>'+case['request']+'<|assistant|>'
        ids = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids], device=device)
        new_ids = []
        started = time.monotonic()
        for _ in range(96):
            next_token = int(net(ids)[:, -1].argmax(-1).item())
            if next_token == eos:
                break
            new_ids.append(next_token)
            ids = torch.cat([ids, torch.tensor([[next_token]], device=device)], dim=1)
            output = tok.decode(new_ids)
            try:
                if isinstance(json.loads(output), dict):
                    break
            except ValueError:
                pass
        output = tok.decode(new_ids)
        try:
            verdict = score(json.loads(output), case)
        except (ValueError, TypeError):
            verdict = {'schema_valid': False, 'correct': False, 'reason': 'Incomplete or invalid JSON'}
        results.append({'id': case['id'], 'split': case['split'], 'request': case['request'],
                        'output': output, 'verification': verdict,
                        'seconds': round(time.monotonic()-started, 3)})
        if len(results) % 8 == 0:
            print(json.dumps({'event': label, 'cases_done': len(results), 'correct': sum(r['verification']['correct'] for r in results)}), flush=True)
    summary = {}
    for split in ('validation', 'audit'):
        subset = [r for r in results if r['split'] == split]
        summary[split] = {'cases': len(subset), 'correct': sum(r['verification']['correct'] for r in subset),
                          'schema_valid': sum(r['verification']['schema_valid'] for r in subset)}
    report = {'label': label, 'decode': 'greedy, float32, 96 tokens maximum, no grammar constraints',
              'summary': summary, 'results': results, 'tools_executed': 0}
    write(out/(label+'.json'), report)
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    original = BRAIN/'trained/brain/best-model.pt'
    saved = torch.load(original, map_location='cpu', weights_only=False)
    net = Brain(Config(**saved['config']))
    net.load_state_dict(saved['model'])
    tok_path = BRAIN/'trained/brain/tokenizer.json'
    tok = Tokenizer.from_file(str(tok_path))
    manifest = read(ROOT/'pilot-data-manifest.json')
    review = read(ROOT/'data-review.json')
    assert hashlib.sha256(tok_path.read_bytes()).hexdigest() == review['tokenizer_sha256']
    assert hashlib.sha256((ROOT/'pilot-train.jsonl').read_bytes()).hexdigest() == manifest['train_sha256']
    train = load_examples(ROOT/'pilot-train.jsonl', tok, net.cfg.context)
    validation = load_examples(ROOT/'sft-validation.jsonl', tok, net.cfg.context)
    cases = [r for r in read(ROOT/'routing-cases.json') if r['split'] != 'train']
    assert len(train) == manifest['training_examples'] and len(validation) == 16 and len(cases) == 32
    pad = tok.token_to_id('<|pad|>')
    if args.check:
        x, y = batch(train[:2], pad, torch.device('cpu'))
        loss = net(x, y)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        print(json.dumps({'event': 'bundle_validated', 'parameters': net.parameter_count(),
                          'training_examples': len(train), 'masked_loss': float(loss.detach()),
                          'finite_backward': True, 'weights_written': False}), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the pilot')
    if (ROOT/'result').exists():
        raise RuntimeError('Existing results will not be overwritten')
    out = ROOT/'result'
    out.mkdir()
    device = torch.device('cuda')
    net.to(device)
    report = {'status': 'baseline', 'started_at': datetime.now(timezone.utc).isoformat(),
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'parameters': net.parameter_count(), 'base_pretraining_tokens': saved.get('tokens'),
              'base_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
              'data': manifest, 'steps': 0, 'optimizer_updates': 0, 'response_tokens_seen': 0,
              'max_training_minutes': 15, 'max_steps': 600, 'batch_size': 8,
              'peak_learning_rate': 1e-4, 'validation': [], 'automatic_promotion': False}
    write(out/'report.json', report)
    before = evaluate(net, tok, cases, device, 'before', out)
    report['language_before'] = regression(net, tok, device)
    initial_loss = validation_loss(net, validation, pad, device, 8)
    report['initial_validation_loss'] = initial_loss
    report['status'] = 'training'
    write(out/'report.json', report)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    best = initial_loss
    started = time.monotonic()
    pool = []
    for step in range(600):
        if time.monotonic()-started > 15*60:
            break
        if len(pool) < 8:
            pool = list(range(len(train)))
            random.shuffle(pool)
        selected = [train[pool.pop()] for _ in range(8)]
        x, y = batch(selected, pad, device)
        rate = 1e-4 * min(1.0, (step+1)/20) * (0.2+0.8*(1+math.cos(math.pi*step/600))/2)
        for group in opt.param_groups:
            group['lr'] = rate
        net.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = net(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        report['steps'] = step+1
        if scaler.get_scale() >= scale_before:
            report['optimizer_updates'] += 1
            report['response_tokens_seen'] += int((y != -100).sum())
        if (step+1) % 50 == 0:
            value = validation_loss(net, validation, pad, device, 8)
            report['validation'].append({'step': step+1, 'loss': value})
            if value < best:
                best = value
                temp = out/'best-model.pt.tmp'
                torch.save({'config': asdict(net.cfg), 'model': {k: v.detach().cpu().half() for k,v in net.state_dict().items()},
                            'base_pretraining_tokens': saved.get('tokens'), 'sft_steps': step+1,
                            'validation_loss': value, 'tokenizer_sha256': review['tokenizer_sha256']}, temp)
                temp.replace(out/'best-model.pt')
                report['selected_step'] = step+1
            report['elapsed_training_seconds'] = round(time.monotonic()-started, 2)
            write(out/'report.json', report)
            print(json.dumps({'event':'training', 'step': step+1, 'validation_loss': value,
                              'seconds': report['elapsed_training_seconds']}), flush=True)
    report['elapsed_training_seconds'] = round(time.monotonic()-started, 2)
    report['peak_gpu_memory_gb'] = torch.cuda.max_memory_allocated()/1e9
    if not (out/'best-model.pt').exists():
        report['status'] = 'no_improving_checkpoint'
        write(out/'report.json', report)
        return
    del opt, scaler
    chosen = torch.load(out/'best-model.pt', map_location='cpu', weights_only=False)
    net.load_state_dict(chosen['model'])
    report['status'] = 'evaluating_candidate'
    write(out/'report.json', report)
    after = evaluate(net, tok, cases, device, 'after', out)
    report['language_after'] = regression(net, tok, device)
    ratio = report['language_after']['weighted_nll']/report['language_before']['weighted_nll']
    gate = manifest['continuation_gate']
    report['before'] = before['summary']
    report['after'] = after['summary']
    report['language_regression_ratio'] = ratio
    report['passes_narrow_pilot_gate'] = (
        after['summary']['audit']['correct'] >= gate['minimum_audit_correct_of_16'] and
        after['summary']['validation']['correct'] >= gate['minimum_validation_correct_of_16'] and
        after['summary']['audit']['schema_valid'] >= gate['minimum_audit_schema_valid_of_16'] and
        ratio <= gate['maximum_regression_nll_ratio'])
    report['status'] = 'complete'
    report['note'] = 'Narrow routing experiment only. Review clarification semantics separately. No autonomous actions and no automatic replacement of the original model.'
    write(out/'report.json', report)
    print(json.dumps({'event':'complete', 'before':report['before'], 'after':report['after'],
                      'regression_ratio':ratio, 'passes_gate':report['passes_narrow_pilot_gate']}), flush=True)

if __name__ == '__main__':
    main()
