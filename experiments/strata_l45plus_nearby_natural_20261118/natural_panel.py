#!/usr/bin/env python3
"""Compare the layer-45 baseline and cumulative layer-46 edit on natural stops."""
from pathlib import Path
import hashlib
import json
import sys
import time
from collections import Counter
from urllib.request import Request, urlopen

OLD=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l45_svd1_20261118')
ROOT=OLD.parent/'strata_l45plus_nearby_natural_20261118'
SOURCE='/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l45-svd1/Qwen3.8-Flash-Next-Gyro-M-Calmer-L45-SVD1-TQ2_0.gguf'
TARGET='/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l45-l46-svd1/Qwen3.8-Flash-Next-Gyro-M-Calmer-L45-L46-SVD1-TQ2_0.gguf'
WRAP='I have thought about this long enough; time to give my answer.'
sys.path.insert(0,str(OLD))
import production_comparison as harness

def freeze():
    p=ROOT/'protocol.json'
    assert not p.exists()
    prior=json.loads((OLD/'production_budget_32768/protocol.json').read_text())
    cases=[c for c in prior['cases'] if c['case'] in ['01','05','07','08','09','10']]
    for c in cases:
        c['request']['max_tokens']=-1
        c['request']['reasoning_budget_tokens']=0
        c['request_sha256']=harness.digest(json.dumps(c['request'],sort_keys=True,separators=(',',':')).encode())
    protocol=dict(created_at=harness.now(),endpoint=prior['endpoint'],cases=cases,
                  objective='Natural completion: cumulative layer45+46 edit versus current layer45 baseline.',
                  selection='Economics and math with both previously frozen seeds, plus both arithmetic controls; chosen before either natural-completion arm.',
                  baseline_model=SOURCE,edited_model=TARGET,
                  output_limit='max_tokens=-1 means remaining 262144-token context, less prompt and engine slack. No smaller test cap.',
                  forced_interventions_disabled=['answer_room','reasoning_budget_tokens','reasoning_close_retry','reasoning_loop_recovery','repeat_stop_tokens'],
                  success='Correct final answer with model stop and no server-inserted wrap-up. Context exhaustion is unfinished, not a successful shortening.',
                  primary_metrics=['natural completion','official-key correctness','completion tokens','wait occurrences','duplicate8_tail'],
                  concurrent_evaluation_requests=1,
                  limitations=['Six requests over two long problems and two arithmetic controls; exploratory, not a benchmark-wide score.',
                               'Current layer45 edit is the baseline; results measure adding the layer46 edit.',
                               'Shared production traffic and speculative decoding permit same-seed variation.',
                               'Medium reasoning instruction remains the existing production policy; it is not a token deadline.'])
    harness.write(p,protocol)
    print(json.dumps({'event':'natural_panel_frozen','cases':[c['case'] for c in cases],'explicit_positive_token_cap':False,'context':262144}),flush=True)

original_identity=harness.identity
original_score=harness.score

def identity(arm):
    d=original_identity(arm)
    cfg=d['config']
    for k,v in [('answer_room',False),('reasoning_budget_tokens',0),('reasoning_close_retry',False),('reasoning_loop_recovery',False),('repeat_stop_tokens',0)]:
        assert cfg.get(k) == v,(k,cfg.get(k))
    assert d['tensor_sha256']=='8974663dc1cda1ebbb63266bc82afc622f63ac3ab38d24a718dcea36539b0c38'
    from gguf_reader import GGUFFile
    path=SOURCE if arm=='baseline' else TARGET
    g=GGUFFile(path)
    t=next(t for t in g.tensors if t.name=='blk.46.ffn_down_shexp.weight')
    with open(path,'rb') as f:
        f.seek(g.data_start+t.offset)
        d['layer46_sha256']=hashlib.sha256(f.read(t.expected_bytes())).hexdigest()
    expected=('4a0375c3811d552220aa2b55c3c5c5f741c96738fe0259be646404918d034749' if arm=='baseline'
              else json.loads((ROOT/'candidate_manifest.json').read_text())['edited_tensor_sha256'])
    assert d['layer46_sha256']==expected
    return d

def score(case,result):
    row=original_score(case,result)
    inserted=WRAP in result['reasoning_content']
    natural=(result.get('done_received',False) and not result.get('error') and
             result['finish_reason']=='stop' and bool(result['content'].strip()) and not inserted)
    row.update(natural_completion=natural,forced_wrap_marker=inserted)
    if not natural and row['status']=='correct':row['status']='unfinished'
    return row

def revise_cap():
    path=ROOT/'protocol.json'
    original=path.read_bytes()
    archive=ROOT/'protocol_before_75000.json'
    assert not archive.exists(), 'Cap revision already recorded'
    protocol=json.loads(original)
    completed=json.loads((ROOT/'baseline/scores.json').read_text())
    assert not completed['complete']
    carried=[r['case'] for r in completed['rows']]
    archive.write_bytes(original)
    interrupted=ROOT/'interrupted_before_75000'
    interrupted.mkdir()
    for case in protocol['cases']:
        if case['case'] not in carried:
            for p in (ROOT/'baseline').glob(case['case']+'.*'):
                p.rename(interrupted/p.name)
        case['request']['max_tokens']=75000
        case['request_sha256']=harness.digest(json.dumps(case['request'],sort_keys=True,separators=(',',':')).encode())
    protocol['revision']=dict(at=harness.now(),reason='User requested a 75000-token failure ceiling.',
        previous_sha256=harness.digest(original),carried_baseline_cases=carried,
        policy='Preserve completed natural stops below 75000. Their original requests used max_tokens=-1; all new requests use 75000. Interrupted attempts remain separate, unscored evidence.')
    protocol['output_limit']='75000 generated tokens, including reasoning and final answer. No forced reasoning closure; a length stop is unfinished.'
    protocol['limitations'].append('Five carried baseline requests used max_tokens=-1 and stopped naturally below 75000. Candidate requests use 75000; these payloads differ only in the inactive output ceiling.')
    harness.write(path,protocol)
    print(json.dumps({'event':'cap_revised','carried':carried,'cap':75000}),flush=True)

def run(arm, protocol_name='protocol.json', output_root=None):
    path=ROOT/protocol_name
    protocol=json.loads(path.read_text())
    protocol_sha=harness.digest(path.read_bytes())
    outdir=(output_root or ROOT)/arm
    outdir.mkdir(parents=True,exist_ok=True)
    current=identity(arm)
    before_path=outdir/'identity_before.json'
    if before_path.exists():
        before=json.loads(before_path.read_text())
        assert before['engines']==current['engines'] and before['config']==current['config']
    else:
        before=current
        harness.write(before_path,before)
    rows=[]
    stopped=False
    for case in protocol['cases']:
        if (outdir/'STOP_AFTER_CURRENT').exists():
            stopped=True
            break
        result_path=outdir/(case['case']+'.result.json')
        if result_path.exists():
            result=json.loads(result_path.read_text())
            actual=json.loads((outdir/(case['case']+'.request.json')).read_text())
            normalized=dict(actual)
            if arm=='baseline' and case['case'] in protocol.get('revision',{}).get('carried_baseline_cases',[]):
                assert normalized['max_tokens']==-1 and result['usage']['completion_tokens']<75000
                normalized['max_tokens']=75000
            assert normalized==case['request'], 'Existing result belongs to a different request'
            rows.append(score(case,result))
            continue
        body=json.dumps(case['request'],sort_keys=True,separators=(',',':')).encode()
        assert harness.digest(body)==case['request_sha256']
        print(json.dumps({'event':'request_start','arm':arm,'case':case['case'],'subject':case['subject'],'seed':case['request']['seed'],'time':harness.now(),'max_tokens':case['request']['max_tokens']}),flush=True)
        harness.write(outdir/(case['case']+'.request.json'),case['request'])
        t0=time.monotonic()
        result=dict(case=case['case'],id=case['id'],request_sha256=case['request_sha256'],started_at=harness.now(),
                    content='',reasoning_content='',finish_reason=None,usage={},timings={},live_before=harness.get('/metrics')['live'])
        try:
            req=Request(protocol['endpoint'],data=body,headers={'Content-Type':'application/json'})
            with urlopen(req,timeout=1200) as response, (outdir/(case['case']+'.events.jsonl')).open('x') as events:
                result['http_status']=response.status
                for line in response:
                    if not line.startswith(b'data:'):continue
                    data=line[5:].strip(); elapsed=time.monotonic()-t0
                    if data==b'[DONE]':
                        result['done_received']=True;break
                    event=json.loads(data)
                    events.write(json.dumps({'elapsed_s':elapsed,'event':event},ensure_ascii=False)+'\n');events.flush()
                    if 'error' in event:raise RuntimeError(str(event['error']))
                    result['response_id']=event.get('id',result.get('response_id'))
                    for choice in event.get('choices',[]):
                        delta=choice.get('delta',{})
                        for key in ['content','reasoning_content']:
                            piece=delta.get(key) or ''
                            result[key]+=piece
                            if piece and 'first_text_s' not in result:result['first_text_s']=elapsed
                        if delta.get('tool_calls'):
                            result.setdefault('tool_call_deltas',[]).extend(delta['tool_calls'])
                        if choice.get('finish_reason'):result['finish_reason']=choice['finish_reason']
                    for key in ['usage','timings']:
                        if event.get(key):result[key]=event[key]
            assert result['finish_reason'] is not None,'Stream ended without finish reason'
            assert result['usage'].get('completion_tokens') is not None,'Missing usage'
        except Exception as error:result['error']=repr(error)
        result.update(elapsed_s=time.monotonic()-t0,ended_at=harness.now())
        harness.write(result_path,result)
        row=score(case,result);rows.append(row)
        harness.write(outdir/'scores.json',dict(arm=arm,protocol_sha256=protocol_sha,rows=rows,
            counts=dict(Counter(r['status'] for r in rows)),complete=len(rows)==len(protocol['cases'])))
        print(json.dumps({'event':'request_complete','arm':arm,**row}),flush=True)
        if result.get('error'):raise RuntimeError(result['error'])
    after=identity(arm)
    assert before['engines']==after['engines'] and before['config']==after['config']
    assert before['tensor_sha256']==after['tensor_sha256'] and before['layer46_sha256']==after['layer46_sha256']
    harness.write(outdir/'identity_after.json',after)
    harness.write(outdir/'scores.json',dict(arm=arm,protocol_sha256=protocol_sha,rows=rows,
        counts=dict(Counter(r['status'] for r in rows)),complete=len(rows)==len(protocol['cases']),stopped_by_user=stopped))
    print(json.dumps({'event':'arm_stopped' if stopped else 'arm_complete','arm':arm,'counts':dict(Counter(r['status'] for r in rows)),
                      'completion_tokens':sum(r['completion_tokens'] for r in rows)}),flush=True)

if __name__=='__main__':
    operation=sys.argv[1]
    harness.ROOT=ROOT
    harness.SOURCE=SOURCE
    harness.TARGET=TARGET
    harness.identity=identity
    harness.score=score
    if operation=='freeze':freeze()
    elif operation=='revise-cap':revise_cap()
    else:
        assert operation in ['baseline','edited']
        run(operation)
