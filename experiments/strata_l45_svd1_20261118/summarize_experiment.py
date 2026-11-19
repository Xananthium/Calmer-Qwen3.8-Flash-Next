#!/usr/bin/env python3
"""Finalize the experiment while leaving the edited production service running."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l45_svd1_20261118')
WRAP='I have thought about this long enough; time to give my answer.'

def read(path):
    return json.loads(path.read_text())

def main():
    run=ROOT/'production_budget_32768'
    scores=read(run/'edited/scores.json')
    protocol=read(run/'protocol.json')
    assert scores['complete'] and len(scores['rows'])==len(protocol['cases'])==10
    assert scores['protocol_sha256']==hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest()
    before=read(run/'edited/identity_before.json')
    after=read(run/'edited/identity_after.json')
    assert before['engines']==after['engines']
    assert before['config']==after['config']
    assert before['router']==after['router']
    baseline_config=read(ROOT/'baseline/identity_before.json')['config']
    normalized=json.loads(json.dumps(after['config']))
    for flag in ['--native','--ple-gguf']:
        index=normalized['args'].index(flag)+1
        normalized['args'][index]=baseline_config['args'][index]
    assert normalized==baseline_config
    rows=[]
    for case,row in zip(protocol['cases'],scores['rows']):
        raw=read(run/'edited'/(case['case']+'.result.json'))
        request=read(run/'edited'/(case['case']+'.request.json'))
        assert request==case['request'] and request['max_tokens']==32768
        assert raw.get('done_received') and not raw.get('error')
        assert raw['request_sha256']==case['request_sha256']
        rows.append({**row,'reasoning_budget_wrap':WRAP in raw['reasoning_content']})
    groups={}
    for name,numbers in [('hle_primary',range(1,7)),('arithmetic_controls',range(7,9)),('hle_second_seed',range(9,11)),('all',range(1,11))]:
        rr=[r for r in rows if int(r['case']) in numbers]
        groups[name]={'n':len(rr),'counts':dict(Counter(r['status'] for r in rr)),
                      'completion_tokens':sum(r['completion_tokens'] for r in rr),
                      'wait_occurrences':sum(r['wait_count'] for r in rr),
                      'reasoning_budget_wraps':sum(r['reasoning_budget_wrap'] for r in rr),
                      'wall_seconds_sum':sum(r['wall_s'] for r in rr)}
    with urlopen('http://127.0.0.1:8003/health',timeout=10) as response:
        health=json.load(response)
    assert health['loaded'] and health['status']=='ok'
    with urlopen('http://127.0.0.1:8001/props',timeout=10) as response:
        props=json.load(response)
    matched=read(ROOT/'comparison.json')
    result={'completed_at':datetime.now(timezone.utc).isoformat(),'deployment':'Edited layer-45 model left running, as requested.',
            'health':health,'engine_pid':after['engines'][0]['pid'],
            'configuration_matches_original_except_model_references':True,
            'production_request_budget':32768,'production_thinking_allowance':24576,
            'production_router_effort':after['router']['effort'],
            'server_default_output_cap':props['default_generation_settings']['params']['n_predict'],
            'production_budget_results':groups,'production_budget_rows':rows,
            'earlier_matched_8192_comparison':matched['groups'],
            'interpretation':['The matched 8192-token experiment did not demonstrate an overall improvement: baseline 7/10 correct, edited 6/10, and edited used more completion tokens.',
                              'The production-budget rerun evaluates the edited model at the observed user request budget. There is no matched unedited 32768-token run, so it cannot establish a weight-edit improvement at that budget.',
                              'Production server configuration was preserved. The mistaken 8192-token cap was in the initial evaluation requests only; changing it to 32768 required no restart.',
                              'Shared production traffic and speculative decoding mean timing and same-seed outputs are not guaranteed deterministic.'],
            'artifacts':{'paired_8192':'comparison.json','production_protocol':'production_budget_32768/protocol.json',
                         'production_results':'production_budget_32768/edited','weight_manifest':'manifest.json',
                         'runtime_proof':'production_settings_verified.json'}}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest=read(ROOT/'manifest.json')
    manifest.update(status='deployed_evaluated_left_running',summary=str(ROOT/'summary.json'),
                    final_health=health,final_engine_pid=result['engine_pid'],user_requested_leave_running=True)
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'production_budget_results':groups,'health':health,'engine_pid':result['engine_pid']},indent=2))

if __name__=='__main__':
    main()
