#!/usr/bin/env python3
"""Compare the current candidate to its preserved layer45-only baseline."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import experiment_l44_l45 as experiment

ROOT=experiment.ROOT;PREVIOUS=experiment.PREVIOUS
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def pair(base_dir,edit_dir,protocol,allow_carried=False):
    b=read(base_dir/'scores.json');e=read(edit_dir/'scores.json')
    assert b['complete'] and e['complete']
    assert len(b['rows'])==len(e['rows'])==len(protocol['cases'])
    carried=set(protocol.get('revision',{}).get('carried_baseline_cases',[])) if allow_carried else set()
    rows=[]
    for br,er in zip(b['rows'],e['rows']):
        assert (br['case'],br['id'],br['seed'])==(er['case'],er['id'],er['seed'])
        req=read(base_dir/(br['case']+'.request.json'));candidate=read(edit_dir/(er['case']+'.request.json'))
        identical=req==candidate
        if br['case'] in carried:
            assert req['max_tokens']==-1 and br['natural_completion'] and br['completion_tokens']<75000
            req['max_tokens']=75000
        assert req==candidate and candidate['max_tokens']==75000 and candidate['reasoning_budget_tokens']==0
        rows.append(dict(case=br['case'],subject=br['subject'],seed=br['seed'],baseline=br,edited=er,
            requests_identical=identical,token_delta=er['completion_tokens']-br['completion_tokens'],
            token_change_pct=100*(er['completion_tokens']/br['completion_tokens']-1),
            retained_natural_correct=br['natural_completion'] and er['natural_completion'] and br['status']==er['status']=='correct'))
    return rows,dict(baseline_protocol_sha256=b['protocol_sha256'],edited_protocol_sha256=e['protocol_sha256'])

def aggregate(rows):
    bt=sum(r['baseline']['completion_tokens'] for r in rows);et=sum(r['edited']['completion_tokens'] for r in rows)
    return dict(n=len(rows),baseline_counts=dict(Counter(r['baseline']['status'] for r in rows)),
        edited_counts=dict(Counter(r['edited']['status'] for r in rows)),baseline_tokens=bt,edited_tokens=et,
        token_change_pct=100*(et/bt-1),
        baseline_natural_completions=sum(r['baseline']['natural_completion'] for r in rows),
        edited_natural_completions=sum(r['edited']['natural_completion'] for r in rows),
        baseline_waits=sum(r['baseline']['wait_count'] for r in rows),edited_waits=sum(r['edited']['wait_count'] for r in rows),
        retained_natural_correct_cases=sum(r['retained_natural_correct'] for r in rows))

def main():
    protocol=read(ROOT/'protocol.json')
    rows,hashes=pair(PREVIOUS/'baseline',ROOT/'edited',protocol,True)
    loop_rows,loop_hashes=pair(PREVIOUS/'loop_probe/baseline',ROOT/'loop_probe/edited',read(ROOT/'loop_protocol.json'))
    bidentity=read(PREVIOUS/'baseline/identity_before.json');eidentity=read(ROOT/'edited/identity_after.json')
    ecfg=json.loads(json.dumps(eidentity['config']))
    for flag in ['--native','--ple-gguf']:
        i=ecfg['args'].index(flag)+1;ecfg['args'][i]=bidentity['config']['args'][i]
    assert ecfg==bidentity['config'] and eidentity['router']==bidentity['router']
    groups={name:aggregate([r for r in rows if r['case'] in ids]) for name,ids in [
        ('long_cases',{'01','05','09','10'}),('arithmetic_controls',{'07','08'}),('all',{r['case'] for r in rows})]}
    loop_groups={s:aggregate([r for r in loop_rows if r['subject']==s]) for s in {r['subject'] for r in loop_rows}}
    old=read(PREVIOUS/'edited/scores.json')
    current={r['case']:r['edited'] for r in rows}
    old_comparison=[dict(case=r['case'],layer45_layer46_tokens=r['completion_tokens'],layer44_layer45_tokens=current[r['case']]['completion_tokens'],
        layer45_layer46_status=r['status'],layer44_layer45_status=current[r['case']]['status']) for r in old['rows'] if r['case'] in current]
    experiment.configure();live=experiment.identity('edited');write(ROOT/'final_live_identity.json',live)
    result=dict(at=datetime.now(timezone.utc).isoformat(),baseline='Layer45 only',edited='Layer44 plus Layer45; Layer46 original',
        **hashes,output_ceiling=75000,forced_endings_disabled=True,context_window=262144,
        groups=groups,rows=rows,loop_probe=dict(**loop_hashes,groups=loop_groups,rows=loop_rows),
        previous_candidate_comparison=dict(scope='Only common completed cases; layer45+46 panel was stopped by user, not a complete six-case run.',rows=old_comparison),
        limitations=protocol['limitations']+read(ROOT/'loop_protocol.json')['limitations'],health=live['health'])
    write(ROOT/'comparison.json',result)
    m=read(ROOT/'candidate_manifest.json');m.update(status='deployed_evaluated_left_running',comparison=str(ROOT/'comparison.json'),final_health=live['health']);write(ROOT/'candidate_manifest.json',m)
    print(json.dumps({'groups':groups,'rows':[{k:r[k] for k in ['case','subject','token_delta','token_change_pct','retained_natural_correct']} for r in rows],'loop_groups':loop_groups,'health':live['health']},indent=2))

if __name__=='__main__':main()
