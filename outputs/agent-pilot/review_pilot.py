"""Record the human-readable review separately from the frozen mechanical score."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    after = json.loads((ROOT/'lightning-result/after.json').read_text(encoding='utf-8'))
    report = json.loads((ROOT/'lightning-result/report.json').read_text(encoding='utf-8'))
    cases = {r['id']: r for r in json.loads((ROOT/'routing-cases.json').read_text(encoding='utf-8'))}
    # These four frozen audit outputs were inspected individually. The original
    # scorer accepts any nonempty clarification; none asks for the missing referent.
    semantic_rejections = {
        'audit-unspecified_reference-0': 'Claims the teacher is unavailable without evidence; does not ask what should be improved.',
        'audit-unspecified_reference-1': 'Assumes a report is missing; should ask which problem and request its description.',
        'audit-unspecified_reference-2': 'Assumes a report should be summarized; should ask which alteration and target.',
        'audit-unspecified_reference-3': 'Claims the teacher is unavailable without evidence; does not ask which options to compare.',
    }
    assert all(any(r['id'] == key and r['verification']['correct'] for r in after['results']) for key in semantic_rejections)
    groups = defaultdict(lambda: {'cases': 0, 'mechanical_correct': 0, 'reviewed_correct': 0})
    rows = []
    for result in after['results']:
        family = cases[result['id']]['family']
        mechanical = result['verification']['correct']
        accepted = mechanical and result['id'] not in semantic_rejections
        for group in (result['split'], family):
            groups[group]['cases'] += 1
            groups[group]['mechanical_correct'] += int(mechanical)
            groups[group]['reviewed_correct'] += int(accepted)
        rows.append({'id': result['id'], 'mechanical_correct': mechanical,
                     'reviewed_correct': accepted, 'semantic_rejection': semantic_rejections.get(result['id'])})
    state = json.loads((ROOT/'lightning-pilot-state.json').read_text(encoding='utf-8'))
    review = {
        'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'method': 'Codex inspected all 32 outputs. Preserve the frozen score; reject four mechanically accepted but irrelevant clarification questions in this separate review.',
        'summary': dict(groups), 'results': rows,
        'decision': 'retain_original_as_principal_archive_candidate_for_experiments',
        'automatic_promotion': False, 'new_gpu_jobs_submitted_by_review': 0,
        'job_cost_credits': state['job_cost'], 'balance_credits': state['balance_latest'],
        'budget_cap_credits': state['budget_credits'],
        'validation_observation': f"Best validation checkpoint: step {report['selected_step']}; later training worsened validation loss.",
        'next_experiment': {
            'scope': 'Improve data before another bounded instruction pilot; do not scale to 1B or spend the remaining balance on this recipe.',
            'data_requirements': [
                'Paired requests with each tool available and unavailable; vary the order of available tools.',
                'Copy complete arithmetic expressions, including parentheses, divisions and multiple operators.',
                'Distinguish conceptual questions about training from requests for the actual project status.',
                'Clarifications must ask for the specific missing input, without inventing tool availability.',
                'Include genuine diverse Portuguese requests and reviewed teacher annotations, beyond template repetition.'
            ],
            'evaluation_policy': 'The old audit is now diagnostic because its failures informed the plan. Freeze a new independent audit before another training run; never claim improvements on the old audit as fresh evidence.',
            'training_policy': 'Restart from the preserved base; validate frequently, use early stopping, select by validation only, then evaluate once on the new audit.',
            'execution_policy': 'Generated routes remain unexecuted; a valid JSON object alone is insufficient evidence of a correct action.'
        }
    }
    (ROOT/'pilot-review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'decision': review['decision'], 'summary': review['summary']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
