"""Two-GPU FP16 pretraining, bounded runs, exact optimizer resume and held-out evaluation."""
import argparse, contextlib, hashlib, json, math, os, shutil, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import Brain, Config

DOMAINS = ['pt', 'en', 'code']
SCHEDULE = ['pt'] * 12 + ['en'] * 5 + ['code'] * 3

def atomic_save(value, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)

def affine_order(n, seed):
    rng = np.random.default_rng(seed)
    a = int(rng.integers(1, max(n, 2)))
    while math.gcd(a, n) != 1:
        a += 1
    return a, int(rng.integers(0, n))

def packed_batch(data, indexes, length, device):
    offsets = indexes[:, None] * length + np.arange(length + 1)[None, :]
    z = torch.from_numpy(np.asarray(data[offsets], dtype=np.int64)).to(device)
    return z[:, :-1], z[:, 1:]

@torch.no_grad()
def evaluate(model, arrays, device, batch=4, batches=8):
    model.eval()
    results = {}
    for domain, data in arrays.items():
        rng = np.random.default_rng(7297)
        total = 0.0
        n = (len(data) - 1) // model.cfg.context
        for _ in range(batches):
            indexes = rng.integers(0, n, size=batch)
            x, y = packed_batch(data, indexes, model.cfg.context, device)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite validation loss')
            total += loss.item()
        results[domain] = total / batches
    results['weighted'] = 0.6 * results['pt'] + 0.25 * results['en'] + 0.15 * results['code']
    model.train()
    return results

def cpu_weights(model):
    weights = {k: v.detach().to(device='cpu', dtype=torch.float16) for k, v in model.state_dict().items() if k != 'head.weight'}
    weights['head.weight'] = weights['embedding.weight']
    return weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='/kaggle/working/brain')
    parser.add_argument('--resume')
    parser.add_argument('--target-tokens', type=int, default=2_000_000_000)
    parser.add_argument('--hours', type=float, default=8.0)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--micro-batch', type=int, default=8)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--save-minutes', type=float, default=25.0)
    args = parser.parse_args()
    started = time.monotonic()
    dist.init_process_group('nccl')
    rank, world = int(os.environ['LOCAL_RANK']), dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.manual_seed(20261003)
    cfg = Config()
    root, out = Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    corpus_hash = hashlib.sha256(manifest_bytes).hexdigest()
    tokenizer_hash = hashlib.sha256((root / 'tokenizer.json').read_bytes()).hexdigest()
    if manifest['vocabulary'] != cfg.vocab_size or tokenizer_hash != manifest['sha256']['tokenizer.json']:
        raise RuntimeError('Tokenizer identity mismatch')
    arrays = {}
    for split in ('train', 'validation', 'audit'):
        arrays[split] = {}
        for domain in DOMAINS:
            path = root / f'{domain}-{split}.bin'
            if path.stat().st_size != manifest['counts'][domain][split] * 2:
                raise RuntimeError('Corpus size mismatch: ' + str(path))
            arrays[split][domain] = np.memmap(path, dtype=np.uint16, mode='r')
    model = Brain(cfg).cuda(rank)
    ddp = DDP(model, device_ids=[rank])
    groups = [{'params': [p for p in model.parameters() if p.ndim >= 2], 'weight_decay': 0.1},
              {'params': [p for p in model.parameters() if p.ndim < 2], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=6e-4, betas=(0.9, 0.95), fused=True)
    scaler = torch.amp.GradScaler('cuda', init_scale=4096)
    step, tokens, best = 0, 0, float('inf')
    counts = {domain: 0 for domain in DOMAINS}
    old_gpu_seconds = 0.0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['config'] != asdict(cfg) or saved['corpus_hash'] != corpus_hash or saved['tokenizer_hash'] != tokenizer_hash:
            raise RuntimeError('Resume identity mismatch')
        if saved['batch_geometry'] != [world, args.micro_batch, args.accumulation]:
            raise RuntimeError('Resume batch geometry mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        step, tokens, counts, best = saved['step'], saved['tokens'], saved['counts'], saved['best_validation']
        old_gpu_seconds = saved.get('cumulative_run_seconds', 0.0)
        del saved
    initial_tokens = tokens
    run_start_step = step
    sizes = {d: (len(arrays['train'][d]) - 1) // cfg.context for d in DOMAINS}
    orders = {d: affine_order(sizes[d], 3000 + i) for i, d in enumerate(DOMAINS)}
    global_tokens = world * args.micro_batch * args.accumulation * cfg.context
    max_tokens = args.target_tokens // global_tokens * global_tokens
    metrics_path = out / 'metrics.jsonl'
    if rank == 0:
        (out / 'config.json').write_text(json.dumps(asdict(cfg), indent=2))
        (out / 'tokenizer.json').write_bytes((root / 'tokenizer.json').read_bytes())
        (out / 'data-manifest.json').write_bytes(manifest_bytes)
        if args.resume:
            previous_best = Path(args.resume).with_name('best-model.pt')
            if not previous_best.exists():
                raise RuntimeError('Best checkpoint missing from previous run')
            shutil.copy2(previous_best, out / 'best-model.pt')

    def log(record):
        if rank == 0:
            record = {'step': step, 'tokens': tokens, 'run_seconds': time.monotonic() - started, **record}
            print(json.dumps(record), flush=True)
            with metrics_path.open('a') as handle:
                handle.write(json.dumps(record) + '\n')

    def save_checkpoint():
        if rank == 0:
            atomic_save({'config': asdict(cfg), 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'scaler': scaler.state_dict(), 'step': step, 'tokens': tokens, 'counts': counts,
                         'best_validation': best, 'corpus_hash': corpus_hash, 'tokenizer_hash': tokenizer_hash,
                         'batch_geometry': [world, args.micro_batch, args.accumulation],
                         'cumulative_run_seconds': old_gpu_seconds + time.monotonic() - started}, out / 'last.pt')
            atomic_save({'config': asdict(cfg), 'model': cpu_weights(model), 'tokens': tokens}, out / 'model.pt')
        dist.barrier()

    def validation(event):
        nonlocal best
        dist.barrier()
        if rank == 0:
            results = evaluate(model, arrays['validation'], rank)
            if results['weighted'] < best:
                best = results['weighted']
                atomic_save({'config': asdict(cfg), 'model': cpu_weights(model), 'tokens': tokens,
                             'validation': results}, out / 'best-model.pt')
            log({'event': event, 'validation': results, 'best_validation': best})
        dist.barrier()

    validation('baseline' if step == 0 else 'resume_validation')
    save_checkpoint()
    last_save = time.monotonic()
    tick = time.monotonic()
    skipped, loss_total = 0, 0.0
    window_steps = 0
    deadline = started + args.hours * 3600 - 180
    while tokens < max_tokens:
        stop = torch.tensor(int(time.monotonic() >= deadline or
                                (args.max_steps is not None and step - run_start_step >= args.max_steps)), device=rank)
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            break
        domain = SCHEDULE[step % len(SCHEDULE)]
        # With independent per-domain counters, every packed block is consumed once
        # before wrapping. Affine permutation shuffles order without an in-memory index.
        n = sizes[domain]
        a, b = orders[domain]
        progress = tokens / max_tokens
        warmup = min(1.0, (tokens + global_tokens) / 20_000_000)
        lr = 6e-4 * warmup if tokens < 20_000_000 else 6e-5 + 0.5 * (6e-4 - 6e-5) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated = torch.zeros((), device=rank)
        for micro in range(args.accumulation):
            start = counts[domain] + (micro * world + rank) * args.micro_batch
            indexes = ((np.arange(start, start + args.micro_batch, dtype=np.int64) % n) * a + b) % n
            x, y = packed_batch(arrays['train'][domain], indexes, cfg.context, rank)
            sync = contextlib.nullcontext() if micro == args.accumulation - 1 else ddp.no_sync()
            with sync:
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = ddp(x, y) / args.accumulation
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                scaler.scale(loss).backward()
            accumulated += loss.detach()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        finite = torch.tensor(int(torch.isfinite(norm)), device=rank)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            # All ranks skip together and lower the scale, retaining the same batch.
            scaler.update(new_scale=scaler.get_scale() / 2)
            skipped += 1
            if skipped >= 8:
                raise RuntimeError('Repeated gradient overflow')
            continue
        scaler.step(optimizer); scaler.update(); skipped = 0
        counts[domain] += world * args.micro_batch * args.accumulation
        step += 1; tokens += global_tokens
        dist.all_reduce(accumulated, op=dist.ReduceOp.SUM)
        loss_total += accumulated.item() / world
        window_steps += 1
        if window_steps == 20:
            torch.cuda.synchronize()
            duration = time.monotonic() - tick
            log({'event': 'train', 'domain': domain, 'loss': loss_total / 20, 'lr': lr,
                 'tokens_per_second': 20 * global_tokens / duration, 'gradient_norm': norm.item()})
            tick = time.monotonic(); loss_total = 0.0; window_steps = 0
        elapsed_save = torch.tensor(int(time.monotonic() - last_save >= args.save_minutes * 60), device=rank)
        dist.all_reduce(elapsed_save, op=dist.ReduceOp.MAX)
        if elapsed_save.item():
            validation('validation'); save_checkpoint(); last_save = time.monotonic(); tick = last_save; loss_total = 0.0
            # Log aggregation windows restart after validation/checkpoint overhead.
            window_steps = 0
    validation('final_validation')
    save_checkpoint()
    if rank == 0:
        report = {'status': 'target_reached' if tokens >= max_tokens else 'checkpoint_saved',
                  'parameters': model.parameter_count(), 'tokens_processed': tokens,
                  'tokens_this_run': tokens - initial_tokens, 'target_tokens': max_tokens,
                  'steps': step, 'best_validation_loss': best, 'training_from_scratch': True,
                  'elapsed_seconds': time.monotonic() - started,
                  'cumulative_run_seconds': old_gpu_seconds + time.monotonic() - started,
                  'corpus_hash': corpus_hash, 'tokenizer_hash': tokenizer_hash,
                  'gpus': [torch.cuda.get_device_name(i) for i in range(world)],
                  'capability_note': 'Base next-token model. Conversational instruction training and autonomous improvement are not yet implemented.'}
        if tokens >= max_tokens:
            candidate = torch.load(out / 'best-model.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(candidate['model'])
            report['selected_checkpoint_tokens'] = candidate['tokens']
            report['audit_of_best'] = evaluate(model, arrays['audit'], rank, batches=16)
        (out / 'report.json').write_text(json.dumps(report, indent=2))
        log({'event': 'finished', **report})
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
