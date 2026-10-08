"""Bounded instruction-tuning recipe. Default: CPU validation without training.

GPU training requires --train and writes a separate candidate. This script does
not launch a cloud machine, manage billing, or promote a candidate automatically.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
BRAIN = ROOT.parent / 'cerebro-v1'
sys.path.insert(0, str(BRAIN))
from model import Brain, Config

def json_read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write_report(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def load_examples(path, tok, context):
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        prompt_ids = tok.encode(item['prompt'], add_special_tokens=False).ids
        answer_ids = tok.encode(item['response']+'<|eos|>', add_special_tokens=False).ids
        joined = prompt_ids + answer_ids
        if not prompt_ids or not answer_ids or len(joined) > context:
            raise ValueError('Invalid or overlong example')
        inputs, labels = joined[:-1], joined[1:]
        labels[:len(prompt_ids)-1] = [-100] * (len(prompt_ids)-1)
        if labels[len(prompt_ids)-1] != answer_ids[0]:
            raise ValueError('First response token is not supervised')
        if sum(value != -100 for value in labels) != len(answer_ids):
            raise ValueError('Unexpected supervision count')
        rows.append({'id': item['id'], 'inputs': inputs, 'labels': labels})
    return rows

def batch(rows, pad, device):
    length = max(len(r['inputs']) for r in rows)
    x = torch.full((len(rows), length), pad, dtype=torch.long)
    y = torch.full((len(rows), length), -100, dtype=torch.long)
    for index, row in enumerate(rows):
        x[index, :len(row['inputs'])] = torch.tensor(row['inputs'])
        y[index, :len(row['labels'])] = torch.tensor(row['labels'])
    return x.to(device), y.to(device)

def validation_loss(net, rows, pad, device, batch_size):
    net.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            x, y = batch(rows[offset:offset+batch_size], pad, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                loss = net(x, y)
            tokens = int((y != -100).sum().item())
            total += float(loss) * tokens
            count += tokens
    net.train()
    return total/count

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', action='store_true')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--minutes', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--out', type=Path, default=ROOT/'sft-candidate')
    args = p.parse_args()
    if not 1 <= args.steps <= 200 or not 1 <= args.minutes <= 30 or not 1 <= args.batch_size <= 8:
        raise ValueError('Recipe limits exceeded')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    folder = BRAIN/'trained/brain'
    original = folder/'best-model.pt'
    saved = torch.load(original, map_location='cpu', weights_only=False)
    tok_path = folder/'tokenizer.json'
    tok = Tokenizer.from_file(str(tok_path))
    review = json_read(ROOT/'data-review.json')
    if hashlib.sha256(tok_path.read_bytes()).hexdigest() != review['tokenizer_sha256']:
        raise ValueError('Tokenizer changed after data review')
    net = Brain(Config(**saved['config']))
    net.load_state_dict(saved['model'])
    train = load_examples(ROOT/'sft-train.jsonl', tok, net.cfg.context)
    validation = load_examples(ROOT/'sft-validation.jsonl', tok, net.cfg.context)
    if {r['id'] for r in train} & {r['id'] for r in validation}:
        raise ValueError('Split overlap')
    pad = tok.token_to_id('<|pad|>')
    if not args.train:
        device = torch.device('cpu')
        x, y = batch(train[:2], pad, device)
        net.eval()
        with torch.no_grad():
            loss = net(x, y)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite masked loss')
        report = {'status': 'cpu_recipe_validated', 'weights_updated': False,
                  'parameters': net.parameter_count(), 'train_examples': len(train),
                  'validation_examples': len(validation), 'audit_examples_seen': 0,
                  'supervised_tokens_in_smoke_batch': int((y != -100).sum().item()),
                  'masked_loss_smoke_batch': float(loss), 'lightning_credits_used': 0,
                  'note': 'One forward pass checks token masking and model compatibility. Training and billing are not tested.'}
        write_report(ROOT/'sft-recipe-validation.json', report)
        print(json.dumps(report), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('--train requires CUDA; no long CPU run is started')
    if args.out.exists():
        raise ValueError('Choose a new candidate directory; existing outputs are preserved')
    args.out.mkdir(parents=True)
    device = torch.device('cuda')
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    initial_loss = validation_loss(net, validation, pad, device, args.batch_size)
    report = {'status': 'running', 'base_checkpoint': str(original),
              'base_checkpoint_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
              'parameters': net.parameter_count(), 'base_validation_loss': initial_loss,
              'steps': 0, 'response_tokens_seen': 0, 'validation': [],
              'limits': {'steps': args.steps, 'training_minutes': args.minutes},
              'automatic_promotion': False,
              'billing_note': 'Training timer excludes upload, setup and validation. Stop the cloud machine separately; this is not a billing cap.'}
    best = initial_loss
    started = time.monotonic()
    for step in range(args.steps):
        if time.monotonic()-started >= args.minutes*60:
            break
        subset = random.sample(train, min(args.batch_size, len(train)))
        x, y = batch(subset, pad, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = net(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        report['steps'] = step+1
        report['response_tokens_seen'] += int((y != -100).sum().item())
        if (step+1) % 25 == 0 or step+1 == args.steps:
            value = validation_loss(net, validation, pad, device, args.batch_size)
            report['validation'].append({'step': step+1, 'loss': value})
            if value < best:
                best = value
                torch.save({'config': asdict(net.cfg), 'model': {k: v.detach().cpu().half() for k, v in net.state_dict().items()},
                            'base_pretraining_tokens': saved.get('tokens'), 'sft_steps': step+1,
                            'validation_loss': value, 'tokenizer_sha256': review['tokenizer_sha256']},
                           args.out/'best-model.pt')
            write_report(args.out/'report.json', report)
            print(json.dumps({'step': step+1, 'validation_loss': value}), flush=True)
    report['status'] = 'complete_candidate_only'
    report['elapsed_training_seconds'] = round(time.monotonic()-started, 2)
    report['best_validation_loss'] = best
    report['candidate_saved'] = (args.out/'best-model.pt').exists()
    write_report(args.out/'report.json', report)

if __name__ == '__main__':
    main()
