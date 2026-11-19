#!/usr/bin/env python3
"""Run the existing natural-stop harness against verified original weights."""
import hashlib,json,sys
from pathlib import Path
PARENT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104')
ROOT=PARENT/'strata_unedited_natural_20261118'
MODEL='/home/researcher/models/qwen38-flash-next-gyro-m/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf'
sys.path.insert(0,str(PARENT/'strata_l45plus_nearby_natural_20261118'))
import natural_panel as n
import loop_panel as loops
natural_score=n.score

def score(case,result):
    row=natural_score(case,result)
    if result.get('interruption'):
        row.update(status='interrupted',natural_completion=False,
            completion_tokens=result['interruption']['live_at_cancellation']['generated'],token_count_is_lower_bound=True,
            token_count_source='Live engine counter immediately before client cancellation; final API usage unavailable.')
    return row

def identity(arm):
    d=n.original_identity(arm);m=json.loads((ROOT/'candidate_manifest.json').read_text())
    for key,value in [('answer_room',False),('reasoning_budget_tokens',0),('reasoning_close_retry',False),('reasoning_loop_recovery',False),('repeat_stop_tokens',0)]:assert d['config'].get(key)==value
    from gguf_reader import GGUFFile
    g=GGUFFile(MODEL)
    with open(MODEL,'rb') as f:
        for layer,expected in m['expected_tensor_hashes'].items():
            t=next(t for t in g.tensors if t.name==f'blk.{layer}.ffn_down_shexp.weight');f.seek(g.data_start+t.offset)
            actual=hashlib.sha256(f.read(t.expected_bytes())).hexdigest();assert actual==expected,(layer,actual,expected)
            d[f'layer{layer}_sha256']=actual
    assert d['tensor_sha256']==m['expected_tensor_hashes']['45']
    return d

def configure():
    n.ROOT=ROOT;n.SOURCE=MODEL;n.TARGET=MODEL;n.identity=identity;n.score=score
    n.harness.ROOT=ROOT;n.harness.SOURCE=MODEL;n.harness.TARGET=MODEL

if __name__=='__main__':
    configure();op=sys.argv[1]
    if op=='identity':print(json.dumps(identity('baseline'),indent=2))
    elif op=='loops':n.score=loops.score;n.run('baseline','loop_protocol.json',ROOT/'loop_probe')
    elif op=='baseline':n.run('baseline')
    else:raise ValueError(op)
