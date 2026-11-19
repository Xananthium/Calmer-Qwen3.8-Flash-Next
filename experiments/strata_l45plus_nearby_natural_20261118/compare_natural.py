#!/usr/bin/env python3
"""Report matched natural-completion runs without rewarding unfinished answers."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import urlopen

ROOT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l45plus_nearby_natural_20261118')

def read(path):
    return json.loads(path.read_text())

def main():
    baseline=read(ROOT/'baseline/scores.json')
    edited=read(ROOT/'edited/scores.json')
    assert baseline['complete'] and edited['complete']
    assert baseline['protocol_sha256']==edited['protocol_sha256']
    bidentity=read(ROOT/'baseline/identity_before.json')
    eidentity=read(ROOT/'edited/identity_before.json')
    bcfg=bidentity['config']
    ecfg=json.loads(json.dumps(eidentity['config']))
    for flag in ['--native','--ple-gguf']:
        index=ecfg['args'].index(flag)+1
        ecfg['args'][index]=bcfg['args'][index]
    assert bcfg==ecfg
    assert bidentity['router']==eidentity['router']
    protocol=read(ROOT/'protocol.json')
    carried=set(protocol.get('revision',{}).get('carried_baseline_cases',[]))
    assert len(baseline['rows'])==len(edited['rows'])==len(protocol['cases'])
    rows=[]
    for b,e in zip(baseline['rows'],edited['rows']):
        assert (b['case'],b['id'],b['seed'])==(e['case'],e['id'],e['seed'])
        request=read(ROOT/'baseline'/(b['case']+'.request.json'))
        candidate_request=read(ROOT/'edited'/(e['case']+'.request.json'))
        identical=request==candidate_request
        if b['case'] in carried:
            assert request['max_tokens']==-1 and b['natural_completion'] and b['completion_tokens']<75000
            request['max_tokens']=75000
        assert request==candidate_request
        assert candidate_request['max_tokens']==75000 and request['reasoning_budget_tokens']==0
        rows.append(dict(case=b['case'],subject=b['subject'],seed=b['seed'],baseline=b,edited=e,
                         request_payloads_identical=identical,
                         token_delta=e['completion_tokens']-b['completion_tokens'],
                         token_change_pct=100*(e['completion_tokens']/b['completion_tokens']-1),
                         retained_natural_correct=(b['natural_completion'] and e['natural_completion'] and b['status']==e['status']=='correct')))
    groups={}
    for name,ids in [('long_cases',{'01','05','09','10'}),('arithmetic_controls',{'07','08'}),('all',{r['case'] for r in rows})]:
        rr=[r for r in rows if r['case'] in ids]
        b_tokens=sum(r['baseline']['completion_tokens'] for r in rr)
        e_tokens=sum(r['edited']['completion_tokens'] for r in rr)
        groups[name]=dict(n=len(rr),baseline_counts=dict(Counter(r['baseline']['status'] for r in rr)),
                          edited_counts=dict(Counter(r['edited']['status'] for r in rr)),
                          baseline_natural_completions=sum(r['baseline']['natural_completion'] for r in rr),
                          edited_natural_completions=sum(r['edited']['natural_completion'] for r in rr),
                          baseline_tokens=b_tokens,edited_tokens=e_tokens,token_change_pct=100*(e_tokens/b_tokens-1),
                          baseline_waits=sum(r['baseline']['wait_count'] for r in rr),edited_waits=sum(r['edited']['wait_count'] for r in rr),
                          retained_natural_correct_cases=sum(r['retained_natural_correct'] for r in rr))
    loops=compare_loops()
    with urlopen('http://127.0.0.1:8003/health',timeout=10) as response:health=json.load(response)
    assert health['loaded'] and health['status']=='ok'
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),baseline='Layer45 edit',edited='Layer45 plus Layer46 edit',
                protocol_sha256=baseline['protocol_sha256'],request_payloads_identical=all(r['request_payloads_identical'] for r in rows),
                carried_baseline_cases=sorted(carried),
                forced_endings_disabled_in_both=True,explicit_test_token_cap=75000,context_window=262144,
                groups=groups,rows=rows,loop_probe=loops,health=health,
                interpretation='Assess token reductions alongside natural completion and correctness. Fewer tokens from a failed or unfinished answer are not a successful reduction.',
                limitations=protocol['limitations']+read(ROOT/'loop_protocol.json')['limitations'])
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    m=read(ROOT/'candidate_manifest.json');m.update(status='deployed_evaluated_left_running',final_health=health,comparison=str(ROOT/'comparison.json'))
    (ROOT/'candidate_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps({'groups':groups,'rows':[{k:r[k] for k in ['case','subject','seed','token_delta','token_change_pct','retained_natural_correct']} for r in rows],'health':health},indent=2))

def compare_loops():
    p=ROOT/'loop_probe'
    b=read(p/'baseline/scores.json');e=read(p/'edited/scores.json')
    assert b['complete'] and e['complete'] and b['protocol_sha256']==e['protocol_sha256']
    assert len(b['rows'])==len(e['rows'])==8
    rows=[]
    for br,er in zip(b['rows'],e['rows']):
        assert (br['case'],br['id'],br['seed'])==(er['case'],er['id'],er['seed'])
        assert (p/'baseline'/(br['case']+'.request.json')).read_bytes()==(p/'edited'/(er['case']+'.request.json')).read_bytes()
        rows.append(dict(case=br['case'],subject=br['subject'],baseline=br,edited=er,
                         token_change_pct=100*(er['completion_tokens']/br['completion_tokens']-1)))
    groups={}
    for subject in sorted({r['subject'] for r in rows}):
        rr=[r for r in rows if r['subject']==subject]
        groups[subject]=dict(n=len(rr),baseline_counts=dict(Counter(r['baseline']['status'] for r in rr)),
            edited_counts=dict(Counter(r['edited']['status'] for r in rr)),
            baseline_tokens=sum(r['baseline']['completion_tokens'] for r in rr),
            edited_tokens=sum(r['edited']['completion_tokens'] for r in rr),
            baseline_repeated_actions=sum(r['baseline'].get('repeated_failed_operation',False) for r in rr),
            edited_repeated_actions=sum(r['edited'].get('repeated_failed_operation',False) for r in rr))
    return dict(protocol_sha256=b['protocol_sha256'],request_payloads_identical=True,groups=groups,rows=rows)

if __name__=='__main__':main()
