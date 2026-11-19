#!/usr/bin/env python3
"""Recovered synthetic loop probes through the unchanged production chat route."""
import json
from pathlib import Path
import re
import sys
from collections import Counter
import natural_panel as n

ROOT=n.ROOT
SOURCE_JOB=n.OLD.parent/'staged_loop_confirmation_api/loop_retention6.generate.json'
SEED_LINE='Wait, I should check that again. The calculation seems right, but perhaps I missed something. Let me go back and verify the same steps before I give the answer.'
base_score=n.score

def freeze():
    path=ROOT/'loop_protocol.json'
    assert not path.exists()
    records=json.loads(SOURCE_JOB.read_text())['records']
    template_request=json.loads((ROOT/'protocol.json').read_text())['cases'][0]['request']
    cases=[]
    for index,record in enumerate(records):
        tool=record.get('kind')=='tool'
        messages=record['messages'] if tool else [
            {'role':'user','content':record['problem']+'\nPlease reason step by step, and put your final answer within \\boxed{}.'},
            {'role':'assistant','content':record['prefix']},
            {'role':'user','content':'Continue exactly where you left off.'}]
        request={**template_request,'messages':messages}
        case=dict(case=f'L{index+1:02d}',id=record['id'],kind='tool' if tool else 'gsm8k',
                  subject='Tool-loop recovery' if tool else 'Repeated-passage recovery',
                  gold=record['expected'] if tool else record['answer'],
                  request=request,source_prompt_sha256=n.harness.digest(record['prompt'].encode()),
                  adaptation='Original message history; rendered by production template.' if tool else
                  'Original problem and five repeated lines as a visible prior assistant message, followed by Continue exactly where you left off. This is not raw unfinished-reasoning prefill.')
        if tool:case['forbidden']=record['forbidden']
        cases.append(case)
    for index,record in enumerate(records[4:],5):
        request={**template_request,'messages':[{'role':'user','content':record['problem']+'\nPlease reason step by step, and put your final answer within \\boxed{}.'}]}
        cases.append(dict(case=f'C{index:02d}',id=record['id']+':clean',kind='gsm8k',subject='Matched clean arithmetic',gold=record['answer'],request=request))
    for case in cases:
        case['request_sha256']=n.harness.digest(json.dumps(case['request'],sort_keys=True,separators=(',',':')).encode())
    protocol=dict(created_at=n.harness.now(),endpoint='http://127.0.0.1:8003/v1/chat/completions',cases=cases,
        source_file=str(SOURCE_JOB),source_sha256=n.harness.digest(SOURCE_JOB.read_bytes()),
        selection='All six recovered synthetic loop cases, plus clean controls for both arithmetic problems. Frozen before this model is tested on these probes.',
        output_limit=75000,concurrent_evaluation_requests=1,
        scoring='One next action or final arithmetic answer. Generated commands are text only and are never executed. Natural stop and correct corrective command/answer required; length stop is unfinished.',
        limitations=['These are reused synthetic development probes, not an independent general benchmark.',
                     'Repeated-passage arithmetic is adapted to chat history; the historical raw-prefill experiment is not reproduced.',
                     'One seed at production sampling; live traffic and speculative decoding can affect replay.'])
    n.harness.write(path,protocol)
    print(json.dumps({'event':'loop_panel_frozen','cases':[c['case'] for c in cases]}),flush=True)

def score(case,result):
    row=base_score(case,result)
    if case['kind']=='tool':
        calls=re.findall(r'<tool_call>(.*?)</tool_call>',result['content'],flags=re.S)
        action='\n'.join(calls)
        commands=[c.strip() for c in re.findall(r'<parameter=command>(.*?)</parameter>',action,flags=re.S)]
        commands += re.findall(r'"command"\s*:\s*"([^"]+)"',action)
        # A backend may expose parsed calls in deltas instead of literal XML.
        grouped={}
        for delta in result.get('tool_call_deltas',[]):
            fn=delta.get('function',{});g=grouped.setdefault(delta.get('index',0),{'name':'','arguments':''})
            g['name']+=fn.get('name','');g['arguments']+=fn.get('arguments','')
        for fn in grouped.values():
            if fn['name']=='shell':
                try:commands.append(json.loads(fn['arguments'])['command'])
                except (ValueError,KeyError,TypeError):pass
        repeated=any(c.strip()==case['forbidden'] for c in commands)
        repeated_op=any(re.match(re.escape(case['forbidden'])+r'(?=$|[\s;&|<>])',c.strip()) is not None for c in commands)
        correct=bool(commands) and any(case['gold'] in c for c in commands) and not repeated_op
        natural=(result.get('done_received',False) and not result.get('error') and
                 result['finish_reason'] in ('stop','tool_calls') and bool(result['content'].strip() or commands) and not row['forced_wrap_marker'])
        status=('error' if result.get('error') else 'cutoff' if result['finish_reason']=='length' else
                'unfinished' if not natural else 'unparsed' if not commands else 'correct' if correct else 'wrong')
        row.update(status=status,predicted='\n'.join(commands) or None,raw_answer='\n'.join(commands),
                   natural_completion=natural,tool_commands=commands,repeated_failed_action=repeated,
                   repeated_failed_operation=repeated_op,direct_corrective_action=correct)
    text=result['reasoning_content']+'\n'+result['content']
    words=text.split();grams=[tuple(words[i:i+8]) for i in range(max(0,len(words)-7))]
    row.update(repeated_seed_line_count=text.count(SEED_LINE),
               duplicate8_full=1-len(set(grams))/len(grams) if grams else 0)
    return row

def main():
    operation=sys.argv[1]
    n.harness.ROOT=ROOT;n.harness.SOURCE=n.SOURCE;n.harness.TARGET=n.TARGET
    if operation=='freeze':freeze()
    else:
        assert operation in ['baseline','edited']
        n.score=score
        n.run(operation,'loop_protocol.json',ROOT/'loop_probe')

if __name__=='__main__':main()
