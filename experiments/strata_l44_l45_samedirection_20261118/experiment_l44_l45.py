#!/usr/bin/env python3
"""Use the same production evaluation client for the layer44+45 candidate."""
import hashlib
import json
from pathlib import Path
import sys

PREVIOUS=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l45plus_nearby_natural_20261118')
ROOT=PREVIOUS.parent/'strata_l44_l45_samedirection_20261118'
sys.path.insert(0,str(PREVIOUS))
import natural_panel as n
import loop_panel as loops

SOURCE='/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l45-svd1/Qwen3.8-Flash-Next-Gyro-M-Calmer-L45-SVD1-TQ2_0.gguf'
TARGET='/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l44-l45/Qwen3.8-Flash-Next-Gyro-M-Calmer-L44-L45-TQ2_0.gguf'

def identity(arm):
    d=n.original_identity(arm)
    for k,v in [('answer_room',False),('reasoning_budget_tokens',0),('reasoning_close_retry',False),('reasoning_loop_recovery',False),('repeat_stop_tokens',0)]:assert d['config'].get(k)==v
    m=json.loads((ROOT/'candidate_manifest.json').read_text())
    assert d['tensor_sha256']==m['layer45_retained_sha256']
    from gguf_reader import GGUFFile
    path=SOURCE if arm=='baseline' else TARGET;g=GGUFFile(path)
    with open(path,'rb') as f:
        for layer in [44,46]:
            t=next(t for t in g.tensors if t.name==f'blk.{layer}.ffn_down_shexp.weight')
            f.seek(g.data_start+t.offset);digest=hashlib.sha256(f.read(t.expected_bytes())).hexdigest()
            expected=m['layer46_restored_sha256'] if layer==46 else m['original_tensor_sha256' if arm=='baseline' else 'edited_tensor_sha256']
            assert digest==expected,(layer,digest,expected)
            d[f'layer{layer}_sha256']=digest
    return d

def configure():
    n.ROOT=ROOT;n.SOURCE=SOURCE;n.TARGET=TARGET
    n.harness.ROOT=ROOT;n.harness.SOURCE=SOURCE;n.harness.TARGET=TARGET
    n.identity=identity

def main():
    configure();operation=sys.argv[1]
    if operation=='identity':print(json.dumps(identity('edited'),indent=2))
    elif operation=='loops':
        n.score=loops.score;n.run('edited','loop_protocol.json',ROOT/'loop_probe')
    elif operation=='edited':n.run('edited')
    else:raise ValueError(operation)

if __name__=='__main__':main()
