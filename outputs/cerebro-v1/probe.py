import json, os, time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import Brain, Config

def main():
    dist.init_process_group('nccl')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.manual_seed(20261003)
    torch.set_num_threads(2)
    net = Brain().cuda(rank)
    params = net.parameter_count()
    ddp = DDP(net, device_ids=[rank])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, fused=True)
    scaler = torch.amp.GradScaler('cuda', init_scale=4096)
    batch, length = 8, net.cfg.context
    x = torch.randint(0, net.cfg.vocab_size, (batch, length + 1), device=rank)
    losses = []
    steps, warmup = 16, 4
    dist.barrier()
    begin = time.monotonic()
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = ddp(x[:, :-1], x[:, 1:])
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        losses.append(loss.item())
        if i == warmup - 1:
            dist.barrier()
            begin = time.monotonic()
    duration = torch.tensor(time.monotonic() - begin, device=rank)
    dist.all_reduce(duration, op=dist.ReduceOp.MAX)
    if rank == 0:
        rate = (steps - warmup) * batch * length * dist.get_world_size() / duration.item()
        report = {'parameters': params, 'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                  'tokens_per_second': rate, 'micro_batch_per_gpu': batch, 'context': length,
                  'peak_memory_gb_per_gpu': torch.cuda.max_memory_allocated() / 1e9,
                  'initial_loss': losses[0], 'last_loss': losses[-1],
                  'estimated_hours_2b_compute_only': 2e9 / rate / 3600,
                  'torch': torch.__version__, 'status': 'PASS'}
        Path('/kaggle/working/probe.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
