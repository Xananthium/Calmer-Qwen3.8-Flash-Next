#!/usr/bin/env python3
"""Reuse the verified four-tensor identity and natural-stop evaluation harness."""
import json,sys
from pathlib import Path
PARENT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104')
ROOT=PARENT/'strata_l2a050_l44_l45_l46a025_20261118'
TARGET='/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l2a050-l44-l45-l46a025/Qwen3.8-Flash-Next-Gyro-M-Calmer-L2A050-L44-L45-L46A025-TQ2_0.gguf'
sys.path.insert(0,str(PARENT/'strata_unedited_natural_20261118'))
import experiment_unedited as shared

def configure():
    shared.ROOT=ROOT;shared.MODEL=TARGET;shared.configure()

def identity(arm):return shared.identity(arm)

if __name__=='__main__':
    configure();op=sys.argv[1]
    if op=='identity':print(json.dumps(identity('edited'),indent=2))
    elif op=='loops':shared.n.score=shared.loops.score;shared.n.run('edited','loop_protocol.json',ROOT/'loop_probe')
    elif op=='edited':shared.n.run('edited')
    else:raise ValueError(op)
