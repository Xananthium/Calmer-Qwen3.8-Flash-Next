#!/usr/bin/env python3
"""Replay saved questions on the live edit with the observed production budget."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l45_svd1_20261118')
sys.path.insert(0,str(ROOT))
import production_comparison as harness

def main():
    destination=ROOT/'production_budget_32768'
    destination.mkdir(exist_ok=False)
    original=ROOT/'protocol.json'
    protocol=json.loads(original.read_text())
    protocol['created_at']=datetime.now(timezone.utc).isoformat()
    protocol['objective']='Replay the same saved baseline questions on the deployed edit using the 32768-token budget observed in the user production logs.'
    protocol['parent_protocol_sha256']=hashlib.sha256(original.read_bytes()).hexdigest()
    protocol['model_condition']='Edited model only. Earlier unedited results used 8192 tokens and are not a matched baseline for this run.'
    protocol['budget_policy']={'max_tokens':32768,'reasoning_allowance':24576,'source':'Existing Strata answer_room default reserves one quarter for the final answer; pre-test production journal entries at 2026-11-18 13:59-14:01 EST show these exact limits.'}
    protocol['limitations'] += ['The 32768-token condition has no matched unedited run; keep its results separate from the earlier 8192-token comparison.']
    for case in protocol['cases']:
        assert case['request']['max_tokens']==8192
        case['request']['max_tokens']=32768
        case['request_sha256']=hashlib.sha256(json.dumps(case['request'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (destination/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'event':'production_budget_frozen','requests':len(protocol['cases']),'total_budget':32768,'reasoning_allowance':24576,'endpoint':protocol['endpoint']}),flush=True)
    harness.ROOT=destination
    harness.run('edited')

if __name__=='__main__':
    main()
