#!/usr/bin/env python3
"""Keep partial stream evidence and let the existing runner skip cancelled math."""
import json
from pathlib import Path

root=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l44_l45_l46_20261118')
out=root/'edited'
assert not (out/'05.result.json').exists()
interruption=json.loads((root/'math_interruption.json').read_text())
case=next(c for c in json.loads((root/'protocol.json').read_text())['cases'] if c['case']=='05')
result=dict(case='05',id=case['id'],request_sha256=case['request_sha256'],
    content='',reasoning_content='',finish_reason=None,usage={},timings={},done_received=False,
    ended_at=interruption['at'],interruption=interruption,error='Administratively cancelled at user request.',
    reconstructed_from_raw_events=True)
for line in (root/'edited.log').read_text().splitlines():
    event=json.loads(line)
    if event.get('case')=='05' and event.get('event')=='request_start':result['started_at']=event['time']
lines=(out/'05.events.jsonl').read_text().splitlines()
for index,line in enumerate(lines):
    try:record=json.loads(line)
    except json.JSONDecodeError:
        assert index==len(lines)-1
        result['truncated_last_event_line']=True
        break
    event=record['event'];result['elapsed_s']=record['elapsed_s']
    result['response_id']=event.get('id',result.get('response_id'))
    for choice in event.get('choices',[]):
        for key in ['content','reasoning_content']:
            piece=choice.get('delta',{}).get(key) or ''
            result[key]+=piece
            if piece and 'first_text_s' not in result:result['first_text_s']=record['elapsed_s']
        assert not choice.get('finish_reason'),'Stream finished before cancellation; review before labeling interrupted.'
    assert not event.get('usage'),'Final usage exists; review before reconstructing interruption.'
assert not result['content'].strip(),'Final content appeared; review interruption.'
(out/'05.result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(case='05',status='interrupted',tokens_observed=interruption['live_at_cancellation']['generated'],final_answer_present=False)))
