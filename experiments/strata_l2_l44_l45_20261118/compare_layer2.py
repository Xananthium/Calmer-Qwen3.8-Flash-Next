#!/usr/bin/env python3
"""Compare the layer2 edit against the completed 44+45 reference."""
from collections import Counter
from datetime import datetime,timezone
import json,sys
from pathlib import Path
import experiment_layer2 as experiment
ROOT=experiment.ROOT;PARENT=ROOT.parent;PREVIOUS=experiment.PREVIOUS
sys.path.insert(0,str(PARENT/'strata_l44_l45_l46_20261118'))
from compare_triplet import pair,aggregate
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    protocol=read(ROOT/'protocol.json');loop_protocol=read(ROOT/'loop_protocol.json')
    rows=pair(PREVIOUS/'edited',ROOT/'edited',protocol)
    loop_rows=pair(PREVIOUS/'loop_probe/edited',ROOT/'loop_probe/edited',loop_protocol)
    b=read(PREVIOUS/'edited/identity_before.json');e=read(ROOT/'edited/identity_after.json')
    cfg=json.loads(json.dumps(e['config']))
    for flag in ['--native','--ple-gguf']:
        i=cfg['args'].index(flag)+1;cfg['args'][i]=b['config']['args'][i]
    assert cfg==b['config'] and b['router']==e['router']
    groups={name:aggregate([r for r in rows if r['case'] in ids]) for name,ids in [
        ('long_cases',{'01','05'}),('arithmetic_controls',{'07','08'}),('all',{'01','05','07','08'})]}
    variants={};loop_variants={}
    paths=[('45 only',PARENT/'strata_l45plus_nearby_natural_20261118/baseline'),
        ('45+46',PARENT/'strata_l45plus_nearby_natural_20261118/edited'),('44+45',PREVIOUS/'edited'),
        ('44+45+46',PARENT/'strata_l44_l45_l46_20261118/edited'),('2+44+45',ROOT/'edited')]
    for label,path in paths:
        for target,p,cases in [(variants,path,protocol['cases']),(loop_variants,path.parent/'loop_probe'/path.name,loop_protocol['cases'])]:
            by={r['case']:r for r in read(p/'scores.json')['rows']};selected=[by[c['case']] for c in cases]
            target[label]=dict(rows=selected,completion_tokens=sum(r['completion_tokens'] for r in selected),
                completion_tokens_are_lower_bound=any(r.get('token_count_is_lower_bound') for r in selected),
                counts=dict(Counter(r['status'] for r in selected)),natural_completions=sum(r['natural_completion'] for r in selected))
    experiment.configure();live=experiment.identity('edited');write(ROOT/'final_live_identity.json',live)
    result=dict(at=datetime.now(timezone.utc).isoformat(),baseline='44+45',edited='2+44+45',request_payloads_identical=True,
        forced_endings_disabled=True,test_output_ceiling=75000,context_window=262144,groups=groups,rows=rows,
        loop_probe=dict(rows=loop_rows,manual_reviews=read(ROOT/'loop_manual_review.json') if (ROOT/'loop_manual_review.json').exists() else []),
        common_case_variants=variants,common_loop_variants=loop_variants,health=live['health'],
        limitations=protocol['limitations']+['45-only historical requests used max_tokens=-1 and stopped below75000; other arms used75000.',
            'The previous44+45+46 math attempt was user-interrupted; its token count is an observed lower bound, with no final answer.'])
    write(ROOT/'comparison.json',result)
    m=read(ROOT/'candidate_manifest.json');m.update(status='deployed_evaluated_left_running',comparison=str(ROOT/'comparison.json'),final_health=live['health']);write(ROOT/'candidate_manifest.json',m)
    print(json.dumps(dict(groups=groups,health=live['health']),indent=2))

if __name__=='__main__':main()
