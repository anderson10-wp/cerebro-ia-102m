"""Verify the downloaded candidate, preserved base and recorded evaluations."""
import hashlib
import json
import sys
import torch
from routing import ROOT, BRAIN, score

sys.path.insert(0, str(BRAIN))
from model import Brain, Config

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4)
    report = json.loads((ROOT/'lightning-result/report.json').read_text(encoding='utf-8'))
    weights = ROOT/'lightning-result/best-model.pt'
    saved = torch.load(weights, map_location='cpu', weights_only=False)
    assert saved['sft_steps'] == report['selected_step'] == 50
    assert saved['validation_loss'] == min(r['loss'] for r in report['validation'])
    assert sha(BRAIN/'trained/brain/best-model.pt') == report['base_sha256']
    assert sha(BRAIN/'trained/brain/tokenizer.json') == saved['tokenizer_sha256']
    assert all(torch.isfinite(t).all() for t in saved['model'].values())
    assert torch.equal(saved['model']['embedding.weight'], saved['model']['head.weight'])
    net = Brain(Config(**saved['config']))
    net.load_state_dict(saved['model'], strict=True)
    assert net.parameter_count() == report['parameters'] == 102257280
    assert net.embedding.weight is net.head.weight
    cases = {c['id']: c for c in json.loads((ROOT/'routing-cases.json').read_text(encoding='utf-8'))}
    after = json.loads((ROOT/'lightning-result/after.json').read_text(encoding='utf-8'))
    for result in after['results']:
        try:
            verdict = score(json.loads(result['output']), cases[result['id']])
        except (ValueError, TypeError):
            verdict = {'schema_valid': False, 'correct': False, 'reason': 'Incomplete or invalid JSON'}
        assert verdict == result['verification']
    evidence = {'status': 'verified', 'candidate_sha256': sha(weights),
                'candidate_bytes': weights.stat().st_size, 'parameters': net.parameter_count(),
                'selected_step': saved['sft_steps'], 'finite_weights': True,
                'original_checkpoint_unchanged': True, 'tokenizer_matches': True,
                'tied_embeddings_equal': True, 'recorded_scores_recomputed': len(after['results']),
                'tools_executed': 0,
                'size_note': 'Serialization duplicates the tied embedding tensors; loading restores sharing. File size does not increase the model parameter count.'}
    (ROOT/'candidate-integrity.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence))

if __name__ == '__main__':
    main()
