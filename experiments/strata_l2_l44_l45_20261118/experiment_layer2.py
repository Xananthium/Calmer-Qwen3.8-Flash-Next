#!/usr/bin/env python3
"""Use the existing production harness for the layer2+44+45 candidate."""
import hashlib,json,sys
from pathlib import Path
PARENT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104')
ROOT=PARENT/'strata_l2_l44_l45_20261118'
PREVIOUS=PARENT/'strata_l44_l45_samedirection_20261118'
sys.path.insert(0,str(PARENT/'strata_l45plus_nearby_natural_20261118'))
import natural_panel as n
import loop_panel as loops
SOURCE='/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l44-l45/Qwen3.8-Flash-Next-Gyro-M-Calmer-L44-L45-TQ2_0.gguf'
TARGET='/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l2-l44-l45/Qwen3.8-Flash-Next-Gyro-M-Calmer-L2-L44-L45-TQ2_0.gguf'
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
    for k,v in [('answer_room',False),('reasoning_budget_tokens',0),('reasoning_close_retry',False),('reasoning_loop_recovery',False),('repeat_stop_tokens',0)]:assert d['config'].get(k)==v
    assert d['tensor_sha256']==m['layer45_retained_sha256']
    from gguf_reader import GGUFFile
    path=SOURCE if arm=='baseline' else TARGET;g=GGUFFile(path)
    expected={2:m['original_tensor_sha256' if arm=='baseline' else 'edited_tensor_sha256'],44:m['layer44_retained_sha256'],46:m['layer46_restored_sha256']}
    with open(path,'rb') as f:
        for layer,digest in expected.items():
            t=next(t for t in g.tensors if t.name==f'blk.{layer}.ffn_down_shexp.weight');f.seek(g.data_start+t.offset)
            actual=hashlib.sha256(f.read(t.expected_bytes())).hexdigest();assert actual==digest,(layer,actual,digest)
            d[f'layer{layer}_sha256']=actual
    return d

def configure():
    n.ROOT=ROOT;n.SOURCE=SOURCE;n.TARGET=TARGET;n.identity=identity;n.score=score
    n.harness.ROOT=ROOT;n.harness.SOURCE=SOURCE;n.harness.TARGET=TARGET

if __name__=='__main__':
    configure();op=sys.argv[1]
    if op=='identity':print(json.dumps(identity('edited'),indent=2))
    elif op=='loops':n.score=loops.score;n.run('edited','loop_protocol.json',ROOT/'loop_probe')
    elif op=='edited':n.run('edited')
    else:raise ValueError(op)
