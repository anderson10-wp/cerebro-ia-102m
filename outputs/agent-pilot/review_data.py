"""Record the explicit review and export supervised examples; does not train."""
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer
from routing import ROOT, BRAIN, SYSTEM, load_json, score, write_json

def format_prompt(case):
    return ('<|system|>' + SYSTEM + '\nDisponíveis: ' + ', '.join(case['available']) +
            '<|user|>' + case['request'] + '<|assistant|>')

def main():
    cases = load_json(ROOT / 'routing-cases.json')
    teacher = load_json(ROOT / 'teacher-routing-results.json')
    by_id = {r['id']: r for r in cases}
    tokenizer_path = BRAIN / 'trained/brain/tokenizer.json'
    tok = Tokenizer.from_file(str(tokenizer_path))
    teacher_by_id = {r['id']: r for r in teacher['results']}
    if len(teacher_by_id) != 32:
        raise ValueError('Expected the complete training-only teacher probe')
    review = {'reviewer': 'Codex', 'teacher_examples_accepted': 0, 'teacher_examples_corrected': 0,
              'reviews': [], 'split_counts': {}, 'maximum_sequence_tokens': 0,
              'weights_updated': False, 'lightning_credits_used': 0}
    fixes = {
        'train-teacher_unavailable-0': 'O professor está indisponível. Você quer aguardar para obter a explicação sobre GPU?',
        'train-teacher_unavailable-1': 'O professor está indisponível. Você quer aguardar para gerar a função de ordenação?',
        'train-teacher_unavailable-2': 'O professor está indisponível. Você quer aguardar para obter a explicação sobre redes neurais?',
        'train-teacher_unavailable-3': 'O professor está indisponível. Você quer aguardar para gerar o programa de leitura de CSV?'}
    exports = {split: [] for split in ('train', 'validation', 'audit')}
    for case in cases:
        route = case['expected']
        provenance = 'independent_case_author'
        if case['split'] == 'train':
            result = teacher_by_id[case['id']]
            result['verification'] = score(result['route'], case)
            if not result['verification']['correct']:
                raise ValueError('Teacher route failed: ' + case['id'])
            route = result['route']
            provenance = 'gpt_oss_teacher_reviewed'
            if case['id'] in fixes:
                route = {'tool': 'clarify', 'arguments': {'question': fixes[case['id']]}}
                provenance = 'gpt_oss_route_with_codex_editor_correction'
                review['teacher_examples_corrected'] += 1
                review['reviews'].append({'id': case['id'], 'accepted': False, 'corrected': True,
                    'reason': 'The original question asked for extra detail instead of explaining that the required teacher was unavailable.'})
            else:
                review['teacher_examples_accepted'] += 1
                review['reviews'].append({'id': case['id'], 'accepted': True,
                    'reason': 'Tool and arguments checked against the independent case; clarification requests ask for the missing artifact.'})
            result['editor_review'] = review['reviews'][-1]
        if not score(route, case)['correct']:
            raise ValueError('Export does not match case')
        prompt = format_prompt(case)
        response = json.dumps(route, ensure_ascii=False, separators=(',', ':'))
        prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
        response_ids = tok.encode(response + '<|eos|>', add_special_tokens=False).ids
        if len(prompt_ids) + len(response_ids) > 1024:
            raise ValueError('Sequence exceeds original model context')
        review['maximum_sequence_tokens'] = max(review['maximum_sequence_tokens'], len(prompt_ids)+len(response_ids))
        exports[case['split']].append({'id': case['id'], 'family': case['family'], 'prompt': prompt,
            'response': response, 'label_source': provenance,
            'loss_policy': 'Mask every prompt token; optimize assistant response and end token only.',
            'prompt_tokens': len(prompt_ids), 'response_tokens': len(response_ids)})
    for split, rows in exports.items():
        (ROOT / f'sft-{split}.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)+'\n', encoding='utf-8')
        review['split_counts'][split] = len(rows)
    review['tokenizer_sha256'] = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    review['training_response_tokens'] = sum(r['response_tokens'] for r in exports['train'])
    review['training_total_tokens'] = sum(r['prompt_tokens'] + r['response_tokens'] for r in exports['train'])
    review['limits'] = ['32 training examples are a seed for a routing experiment, not a complete instruction-tuning dataset.',
                       '16 validation and 16 audit cases are too small to establish broad competence.',
                       'Clarification correctness in automated scoring only measures routing/schema; language meaning was reviewed separately for training examples.',
                       'No tools or autonomous training are enabled in the 102M by these exports.']
    teacher['status'] = 'reviewed'
    teacher['normalized_expression_rule'] = 'Ignore arithmetic whitespace only; preserve numbers and operators.'
    write_json(ROOT / 'teacher-routing-results.json', teacher)
    write_json(ROOT / 'data-review.json', review)
    write_json(ROOT / 'proposal-review.json', {
        'original_plan_format_valid': True, 'original_plan_approved_for_execution': False,
        'reasons': ['The predicted 5% gain has no empirical basis and no training intervention.',
                    'The CPU latency threshold of 2 seconds was not measured or justified.',
                    'Accuracy and regression thresholds were invented rather than agreed and tied to a specified test.'],
        'replacement_next_step': 'Record the frozen scratch-102M baseline; review training-only teacher examples; expand the instruction dataset before proposing a capped GPU run.',
        'principal_model': 'scratch_102m', 'teacher': 'gpt-oss-20b', 'execute_training': False})
    print(json.dumps({k: review[k] for k in ('teacher_examples_accepted', 'teacher_examples_corrected',
                    'split_counts', 'maximum_sequence_tokens', 'training_response_tokens', 'training_total_tokens')}, ensure_ascii=True))

if __name__ == '__main__':
    main()
