#!/usr/bin/env python3
"""Add a matched original-weight baseline to the complete experiment table."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import experiment_unedited as experiment
ROOT=experiment.ROOT;PARENT=ROOT.parent
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    protocol=read(ROOT/'protocol.json');loop_protocol=read(ROOT/'loop_protocol.json')
    baseline=read(ROOT/'baseline/scores.json');assert baseline['complete'] and len(baseline['rows'])==len(protocol['cases'])
    baseline_by={r['case']:r for r in baseline['rows']}
    b_identity=read(ROOT/'baseline/identity_after.json')
    variants={};loop_variants={};comparisons={};request_checks={}
    paths=[('Unedited',ROOT/'baseline'),('45 only',PARENT/'strata_l45plus_nearby_natural_20261118/baseline'),
        ('45+46',PARENT/'strata_l45plus_nearby_natural_20261118/edited'),
        ('44+45',PARENT/'strata_l44_l45_samedirection_20261118/edited'),
        ('44+45+46',PARENT/'strata_l44_l45_l46_20261118/edited'),
        ('2+44+45',PARENT/'strata_l2_l44_l45_20261118/edited')]
    for label,path in paths:
        cfg_identity=read(path/'identity_before.json');cfg=json.loads(json.dumps(cfg_identity['config']))
        for flag in ['--native','--ple-gguf']:
            i=cfg['args'].index(flag)+1;cfg['args'][i]=b_identity['config']['args'][i]
        assert cfg==b_identity['config'] and cfg_identity['router']==b_identity['router'],label
        checks=[]
        for family,out,p,cases in [('core',variants,path,protocol['cases']),('loops',loop_variants,path.parent/'loop_probe'/path.name,loop_protocol['cases'])]:
            by={r['case']:r for r in read(p/'scores.json')['rows']};selected=[]
            for case in cases:
                row=by[case['case']];assert (row['id'],row['seed'])==(case['id'],case['request']['seed'])
                req=read(p/(case['case']+'.request.json'));normalized=dict(req)
                if label=='45 only' and normalized['max_tokens']==-1:
                    assert row['natural_completion'] and row['completion_tokens']<75000
                    normalized['max_tokens']=75000
                    checks.append(dict(family=family,case=case['case'],difference='Historical max_tokens=-1; natural stop below75000.'))
                assert normalized==case['request'],(label,family,case['case'])
                selected.append(row)
            out[label]=dict(rows=selected,completion_tokens=sum(r['completion_tokens'] for r in selected),
                completion_tokens_are_lower_bound=any(r.get('token_count_is_lower_bound') for r in selected),
                counts=dict(Counter(r['status'] for r in selected)),natural_completions=sum(r['natural_completion'] for r in selected))
        request_checks[label]=dict(config_same_except_model_paths=True,router_identical=True,request_differences=checks)
        paired=[]
        for row in variants[label]['rows']:
            b=baseline_by[row['case']];natural_pair=b['natural_completion'] and row['natural_completion']
            paired.append(dict(case=row['case'],baseline=b,variant=row,natural_pair=natural_pair,
                both_correct_and_natural=natural_pair and b['status']==row['status']=='correct',
                token_change_pct=100*(row['completion_tokens']/b['completion_tokens']-1) if natural_pair else None))
        comparisons[label]=paired
    experiment.configure();live=experiment.identity('baseline');write(ROOT/'final_live_identity.json',live)
    result=dict(at=datetime.now(timezone.utc).isoformat(),baseline='Original unedited GGUF weights under unchanged production steering',
        runtime_control_vector_retained=True,forced_endings_disabled=True,test_output_ceiling=75000,context_window=262144,
        common_case_variants=variants,common_loop_variants=loop_variants,comparisons_to_unedited=comparisons,
        verification=request_checks,earlier_baseline_audit=read(ROOT/'earlier_baseline_audit.json'),
        manual_reviews=read(ROOT/'manual_reviews.json') if (ROOT/'manual_reviews.json').exists() else [],
        historical_loop_manual_reviews=read(PARENT/'strata_l44_l45_l46_20261118/loop_manual_review.json'),
        health=live['health'],limitations=protocol['limitations']+['45-only historical requests used no positive cap, but naturally ended below75000.',
            'The44+45+46 math result is user-interrupted, with an observed lower-bound token count and no final answer.'])
    write(ROOT/'comparison.json',result)
    m=read(ROOT/'candidate_manifest.json');m.update(status='deployed_evaluated_left_running',comparison=str(ROOT/'comparison.json'),final_health=live['health']);write(ROOT/'candidate_manifest.json',m)
    print(json.dumps(dict(core={k:{x:y for x,y in v.items() if x!='rows'} for k,v in variants.items()},health=live['health']),indent=2))

if __name__=='__main__':main()
