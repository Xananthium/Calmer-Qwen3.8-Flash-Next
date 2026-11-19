#!/usr/bin/env python3
"""Extend the verified historical table with the partial2/46 experiment."""
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,sys
from pathlib import Path
import experiment_partial_2_46 as experiment
ROOT=experiment.ROOT;PARENT=ROOT.parent
UNEDITED=PARENT/'strata_unedited_natural_20261118'
BEST=PARENT/'strata_l44_l45_samedirection_20261118'
sys.path.insert(0,str(PARENT/'strata_l44_l45_l46_20261118'))
from compare_triplet import pair,aggregate
LABEL='44+45 + 2:50% + 46:25%'
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    protocol=read(ROOT/'protocol.json');loop_protocol=read(ROOT/'loop_protocol.json')
    comparisons={label:pair(path,ROOT/'edited',protocol) for label,path in [('Unedited',UNEDITED/'baseline'),('44+45',BEST/'edited')]}
    loop_rows=pair(UNEDITED/'loop_probe/baseline',ROOT/'loop_probe/edited',loop_protocol)
    reference=read(UNEDITED/'baseline/identity_after.json');current=read(ROOT/'edited/identity_after.json')
    cfg=json.loads(json.dumps(current['config']))
    for flag in ['--native','--ple-gguf']:
        i=cfg['args'].index(flag)+1;cfg['args'][i]=reference['config']['args'][i]
    assert cfg==reference['config'] and current['router']==reference['router']
    result=read(UNEDITED/'comparison.json')
    for key,path,cases in [('common_case_variants',ROOT/'edited',protocol['cases']),('common_loop_variants',ROOT/'loop_probe/edited',loop_protocol['cases'])]:
        scores=read(path/'scores.json');assert scores['complete'];by={r['case']:r for r in scores['rows']}
        rows=[by[c['case']] for c in cases]
        result[key][LABEL]=dict(rows=rows,completion_tokens=sum(r['completion_tokens'] for r in rows),
            completion_tokens_are_lower_bound=any(r.get('token_count_is_lower_bound') for r in rows),
            counts=dict(Counter(r['status'] for r in rows)),natural_completions=sum(r['natural_completion'] for r in rows))
    groups={label:{name:aggregate([r for r in rows if r['case'] in ids]) for name,ids in [
        ('all',{'01','05','07','08'}),('long_cases',{'01','05'}),('arithmetic_controls',{'07','08'})]} for label,rows in comparisons.items()}
    experiment.configure();live=experiment.identity('edited');write(ROOT/'final_live_identity.json',live)
    result.update(at=datetime.now(timezone.utc).isoformat(),additional_candidate=LABEL,partial_strengths={'2':0.5,'46':0.25},
        candidate_comparisons=comparisons,candidate_groups=groups,candidate_loop_vs_unedited=loop_rows,
        source_comparison=str(UNEDITED/'comparison.json'),source_comparison_sha256=hashlib.sha256((UNEDITED/'comparison.json').read_bytes()).hexdigest(),health=live['health'])
    result['verification'][LABEL]=dict(config_same_except_model_paths=True,router_identical=True,request_differences=[])
    result['limitations'].append('Partial2 and46 were changed together; this run evaluates their combination on44+45.')
    if (ROOT/'manual_reviews.json').exists():result['manual_reviews']+=read(ROOT/'manual_reviews.json')
    write(ROOT/'comparison.json',result)
    m=read(ROOT/'candidate_manifest.json');m.update(status='deployed_evaluated_left_running',comparison=str(ROOT/'comparison.json'),final_health=live['health']);write(ROOT/'candidate_manifest.json',m)
    print(json.dumps(dict(groups=groups,health=live['health']),indent=2))

if __name__=='__main__':main()
