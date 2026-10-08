import json, sys, tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path('outputs/cerebro-v1').resolve()))
from model import Brain, Config
from prepare import fingerprint, split_for, source_files
from train import affine_order, packed_batch

torch.set_num_threads(2)
torch.manual_seed(17)
small = Config(vocab_size=256, dim=64, layers=2, heads=4, hidden_dim=128, context=32)
net = Brain(small)
x = torch.randint(0, 256, (2, 16))
net.eval()
with torch.no_grad():
    original = net(x)
    changed = x.clone(); changed[:, 8:] = torch.randint(0, 256, (2, 8))
    assert torch.allclose(original[:, :8], net(changed)[:, :8], atol=1e-6), 'Causal mask leakage'
opt = torch.optim.AdamW(net.parameters(), lr=0.005)
target = x.roll(-1, 1)
initial = net(x, target).item()
for _ in range(40):
    opt.zero_grad(); loss = net(x, target); loss.backward(); opt.step()
final = net(x, target).item()
assert final < initial * 0.5, (initial, final)
assert net.embedding.weight.data_ptr() == net.head.weight.data_ptr(), 'Embedding weights not tied'
with tempfile.TemporaryDirectory(dir='work') as temp:
    path = Path(temp)/'checkpoint.pt'
    torch.save(net.state_dict(), path)
    restored = Brain(small)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.allclose(net(x), restored(x), atol=1e-6)
for n in [7, 128, 513]:
    a, b = affine_order(n, 15)
    assert len(set(((np.arange(n)*a+b)%n).tolist())) == n
assert fingerprint('a b\nc') == fingerprint('a   b c')
splits = {split_for(fingerprint(str(i))) for i in range(10000)}
assert splits == {'train', 'validation', 'audit'}
data=np.arange(257,dtype=np.uint16)
bx,by=packed_batch(data,np.array([0,7]),32,'cpu')
assert torch.equal(bx[:,1:],by[:,:-1])
full = Brain()
assert full.parameter_count() == 102257280
report={'status':'PASS','parameters':full.parameter_count(),'causality':'PASS','checkpoint_round_trip':'PASS',
        'unique_block_permutation':'PASS','document_split':'PASS','tiny_training_initial_loss':initial,'tiny_training_final_loss':final}
Path('outputs/cerebro-v1/local-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
