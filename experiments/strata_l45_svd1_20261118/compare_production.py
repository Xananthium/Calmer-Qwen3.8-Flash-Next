#!/usr/bin/env python3
"""Compare the frozen baseline and edited production runs, retaining all cases."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l45_svd1_20261118')
WRAP_MARKER = 'I have thought about this long enough; time to give my answer.'

def read(name):
    return json.loads((ROOT/name).read_text())

def main():
    baseline, edited = read('baseline/scores.json'), read('edited/scores.json')
    assert baseline['complete'] and edited['complete']
    assert baseline['protocol_sha256'] == edited['protocol_sha256']
    before, after = read('baseline/identity_before.json'), read('edited/identity_before.json')
    bcfg, ecfg = before['config'], after['config']
    args = ecfg['args']
    native_index = args.index('--native')+1
    edited_native = args[native_index]
    for flag in ['--native','--ple-gguf']:
        index = args.index(flag)+1
        args[index] = bcfg['args'][index]
    assert bcfg == ecfg, 'Unexpected serving configuration difference'
    rows = []
    for b, e in zip(baseline['rows'], edited['rows']):
        assert (b['case'],b['id'],b['seed']) == (e['case'],e['id'],e['seed'])
        assert (ROOT/'baseline'/(b['case']+'.request.json')).read_bytes() == (ROOT/'edited'/(e['case']+'.request.json')).read_bytes()
        b_raw=read('baseline/'+b['case']+'.result.json')
        e_raw=read('edited/'+e['case']+'.result.json')
        rows.append(dict(case=b['case'], id=b['id'], subject=b['subject'], seed=b['seed'],
                         baseline=b, edited=e, completion_token_delta=e['completion_tokens']-b['completion_tokens'],
                         baseline_budget_wrap=WRAP_MARKER in b_raw['reasoning_content'],
                         edited_budget_wrap=WRAP_MARKER in e_raw['reasoning_content'],
                         correctness_change=('gain' if b['status']!='correct' and e['status']=='correct' else
                                             'loss' if b['status']=='correct' and e['status']!='correct' else 'unchanged')))
    groups = {}
    for name, numbers in [('hle_primary',range(1,7)),('arithmetic_controls',range(7,9)),('hle_second_seed',range(9,11)),('all',range(1,11))]:
        rr = [r for r in rows if int(r['case']) in numbers]
        b_tokens=sum(r['baseline']['completion_tokens'] for r in rr)
        e_tokens=sum(r['edited']['completion_tokens'] for r in rr)
        groups[name] = dict(n=len(rr), baseline_counts=dict(Counter(r['baseline']['status'] for r in rr)),
                            edited_counts=dict(Counter(r['edited']['status'] for r in rr)),
                            correctness_changes=dict(Counter(r['correctness_change'] for r in rr)),
                            baseline_completion_tokens=b_tokens, edited_completion_tokens=e_tokens,
                            completion_token_change_pct=100*(e_tokens/b_tokens-1),
                            baseline_reasoning_budget_wraps=sum(r['baseline_budget_wrap'] for r in rr),
                            edited_reasoning_budget_wraps=sum(r['edited_budget_wrap'] for r in rr),
                            baseline_wait_count=sum(r['baseline']['wait_count'] for r in rr),
                            edited_wait_count=sum(r['edited']['wait_count'] for r in rr))
    result=dict(protocol_sha256=baseline['protocol_sha256'], request_payloads_identical=True,
                only_serving_configuration_change='The --native and --ple-gguf references point to the edited copy of the same model', edited_native=edited_native,
                baseline_tensor_sha256=before['tensor_sha256'], edited_tensor_sha256=after['tensor_sha256'],
                groups=groups, rows=rows,
                backend_budget_policy=dict(total_request_budget=8192,reasoning_budget=6144,
                                           cause='Existing Strata answer_room default reserves the final quarter for the answer.',
                                           source='src/serve/server.py:6350-6357 and REASONING_WRAP_UP at line 233',
                                           applied_equally_to_both_arms=True,
                                           interpretation='A stop finish does not imply uncapped reasoning; forced wrap-ups are counted separately.'),
                limitations=read('protocol.json')['limitations'])
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'groups':groups,'request_payloads_identical':True,'rows':[
        {k:r[k] for k in ['case','subject','seed','correctness_change','completion_token_delta']} for r in rows]},indent=2))

if __name__=='__main__':
    main()
