#!/usr/bin/env python3
"""Report the common core cases without pooling incomplete experiments."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import experiment_triplet as experiment
ROOT=experiment.ROOT;PREVIOUS=experiment.PREVIOUS;PARENT=ROOT.parent
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def pair(bdir,edir,protocol):
    b=read(bdir/'scores.json');e=read(edir/'scores.json')
    assert e['complete'] and len(e['rows'])==len(protocol['cases'])
    bm={r['case']:r for r in b['rows']};em={r['case']:r for r in e['rows']};rows=[]
    for c in protocol['cases']:
        br,er=bm[c['case']],em[c['case']]
        assert (br['id'],br['seed'])==(er['id'],er['seed'])
        req=read(bdir/(c['case']+'.request.json'));candidate=read(edir/(c['case']+'.request.json'))
        assert req==candidate and req['max_tokens']==75000 and req['reasoning_budget_tokens']==0
        rows.append(dict(case=c['case'],subject=c['subject'],baseline=br,edited=er,
            token_delta=er['completion_tokens']-br['completion_tokens'],
            token_change_pct=None if er.get('token_count_is_lower_bound') else 100*(er['completion_tokens']/br['completion_tokens']-1),
            token_delta_is_lower_bound=er.get('token_count_is_lower_bound',False),
            retained_natural_correct=br['natural_completion'] and er['natural_completion'] and br['status']==er['status']=='correct'))
    return rows

def aggregate(rows):
    bt=sum(r['baseline']['completion_tokens'] for r in rows);et=sum(r['edited']['completion_tokens'] for r in rows)
    censored=any(r['edited'].get('token_count_is_lower_bound') for r in rows)
    return dict(n=len(rows),baseline_counts=dict(Counter(r['baseline']['status'] for r in rows)),
        edited_counts=dict(Counter(r['edited']['status'] for r in rows)),baseline_tokens=bt,edited_tokens=et,
        token_change_pct=None if censored else 100*(et/bt-1),edited_tokens_are_lower_bound=censored,baseline_natural_completions=sum(r['baseline']['natural_completion'] for r in rows),
        edited_natural_completions=sum(r['edited']['natural_completion'] for r in rows),
        baseline_waits=sum(r['baseline']['wait_count'] for r in rows),edited_waits=sum(r['edited']['wait_count'] for r in rows))

def main():
    protocol=read(ROOT/'protocol.json');rows=pair(PREVIOUS/'edited',ROOT/'edited',protocol)
    loop_rows=pair(PREVIOUS/'loop_probe/edited',ROOT/'loop_probe/edited',read(ROOT/'loop_protocol.json'))
    b=read(PREVIOUS/'edited/identity_before.json');e=read(ROOT/'edited/identity_after.json')
    cfg=json.loads(json.dumps(e['config']))
    for flag in ['--native','--ple-gguf']:
        i=cfg['args'].index(flag)+1;cfg['args'][i]=b['config']['args'][i]
    assert cfg==b['config'] and b['router']==e['router']
    groups={name:aggregate([r for r in rows if r['case'] in ids]) for name,ids in [
        ('long_cases',{'01','05'}),('arithmetic_controls',{'07','08'}),('all',{r['case'] for r in rows})]}
    variants={};loop_variants={}
    loop_cases=read(ROOT/'loop_protocol.json')['cases']
    for label,path in [('45 only',PARENT/'strata_l45plus_nearby_natural_20261118/baseline'),
                       ('45+46',PARENT/'strata_l45plus_nearby_natural_20261118/edited'),
                       ('44+45',PREVIOUS/'edited'),('44+45+46',ROOT/'edited')]:
        by={r['case']:r for r in read(path/'scores.json')['rows']}
        selected=[by[c['case']] for c in protocol['cases']]
        variants[label]=dict(rows=selected,completion_tokens=sum(r['completion_tokens'] for r in selected),
            completion_tokens_are_lower_bound=any(r.get('token_count_is_lower_bound') for r in selected),
            counts=dict(Counter(r['status'] for r in selected)),natural_completions=sum(r['natural_completion'] for r in selected))
        loop_by={r['case']:r for r in read(path.parent/'loop_probe'/path.name/'scores.json')['rows']}
        loop_selected=[loop_by[c['case']] for c in loop_cases]
        loop_variants[label]=dict(rows=loop_selected,completion_tokens=sum(r['completion_tokens'] for r in loop_selected),
            counts=dict(Counter(r['status'] for r in loop_selected)),natural_completions=sum(r['natural_completion'] for r in loop_selected))
    experiment.configure();live=experiment.identity('edited');write(ROOT/'final_live_identity.json',live)
    result=dict(at=datetime.now(timezone.utc).isoformat(),baseline='44+45',edited='44+45+46',request_payloads_identical=True,
        forced_endings_disabled=True,test_output_ceiling=75000,context_window=262144,groups=groups,rows=rows,
        loop_probe=dict(groups={s:aggregate([r for r in loop_rows if r['subject']==s]) for s in {r['subject'] for r in loop_rows}},rows=loop_rows,
            manual_reviews=read(ROOT/'loop_manual_review.json') if (ROOT/'loop_manual_review.json').exists() else []),
        common_case_variants=variants,common_loop_variants=loop_variants,health=live['health'],
        interruptions=[read(ROOT/'math_interruption.json')] if (ROOT/'math_interruption.json').exists() else [],
        limitations=protocol['limitations']+['45-only historical requests used max_tokens=-1 and naturally ended below75000; other arms used75000. This four-case table is not the full earlier panel.',
            'User-interrupted requests have no final answer or final API usage; their token counts are observed lower bounds, never successful shortening.'])
    write(ROOT/'comparison.json',result)
    m=read(ROOT/'candidate_manifest.json');m.update(status='deployed_evaluated_left_running',comparison=str(ROOT/'comparison.json'),final_health=live['health']);write(ROOT/'candidate_manifest.json',m)
    print(json.dumps({'groups':groups,'common_case_variants':{k:{x:y for x,y in v.items() if x!='rows'} for k,v in variants.items()},'health':live['health']},indent=2))

if __name__=='__main__':main()
