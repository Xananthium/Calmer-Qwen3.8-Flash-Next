#!/usr/bin/env python3
"""Frozen paired evaluation through the live production reasoning router."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.request import Request, urlopen

PARENT = Path('/home/researcher/model-tools/qwen38-weight-edit-20261104')
ROOT = PARENT/'strata_l45_svd1_20261118'
STRATA = Path('/home/researcher/strata-122b')
BASE_URL = 'http://127.0.0.1:8003'
SOURCE = '/home/researcher/models/qwen38-flash-next-gyro-m/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf'
TARGET = '/home/researcher/models/qwen38-flash-next-gyro-m-calmer-l45-svd1/Qwen3.8-Flash-Next-Gyro-M-Calmer-L45-SVD1-TQ2_0.gguf'

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def get(route):
    with urlopen(BASE_URL+route, timeout=20) as response:
        return json.load(response)

def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT/'protocol.json'
    assert not target.exists(), 'Frozen protocol already exists'
    development_path = PARENT/'staged_confirmation_api_16k/development7.generate.json'
    reserve_path = PARENT/'staged_confirmation_api_16k/hle_reserve8.generate.json'
    development = json.loads(development_path.read_text())['records']
    reserve = json.loads(reserve_path.read_text())['records']
    selected = [development[i] for i in [0,1,3]] + [reserve[i] for i in [4,6,7]] + development[4:6]
    cfg = json.loads((STRATA/'fn-iq3s/cfg-g4x.json').read_text())
    cases = []
    for index, record in enumerate(selected):
        messages = record.get('messages') or [{'role':'user','content':record['problem']+'\nPlease reason step by step, and put your final answer within \\boxed{}.'}]
        request = dict(model=cfg['model_name'], messages=messages, **cfg['sampling'],
                       seed=20261118, reasoning_effort='medium', max_tokens=8192,
                       stream=True, stream_options={'include_usage':True})
        cases.append(dict(case=f'{index+1:02d}', id=record['id'], kind=record.get('kind','gsm8k'),
                          subject=record.get('subject','Arithmetic'), gold=record['answer'],
                          choices=record.get('choices'), request=request))
    for index in [0,4]:
        case = json.loads(json.dumps(cases[index]))
        case['case'] = f'{len(cases)+1:02d}'
        case['request']['seed'] = 20261119
        cases.append(case)
    for case in cases:
        case['request_sha256'] = digest(json.dumps(case['request'],sort_keys=True,separators=(',',':')).encode())
    protocol = dict(created_at=now(), endpoint=BASE_URL+'/v1/chat/completions',
                    objective='Paired baseline versus layer-45 shared-expert rank-one deflation in the same production Strata backend',
                    selection='Fixed before any current-model results: development indices 0,1,3; reserve indices 4,6,7; development arithmetic indices 4,5; second-seed repeats for economics and mathematics.',
                    source_files={str(p):digest(p.read_bytes()) for p in [development_path,reserve_path]},
                    cases=cases, concurrent_evaluation_requests=1,
                    metrics=['official-key correctness','cutoffs','unparsed answers','completion tokens','duplicate tail 8-grams','wait count (descriptive only)','wall time and server timings'],
                    limitations=['Small exploratory diagnostic, not a full HLE benchmark.','Shared live production load and cache can affect timing and stochastic replay.','Repeated seeds assess only limited sampling variation.','Publicly emitted reasoning and final answers are saved; reference keys are never sent to the model.'],
                    runtime='Current temperature 1.3, top_p .95, top_k 20, medium reasoning router, MTP, control vector and all engine settings preserved.',
                    continuation_policy='No hidden retries or continuation. Length stops remain cutoffs; any later extension is a separately documented matched run.')
    write(target,protocol)
    print(json.dumps({'event':'frozen','path':str(target),'requests':len(cases),'sha256':digest(target.read_bytes())}),flush=True)

def identity(arm):
    env = (STRATA/'fn-iq3s/boot.env').read_text().strip()
    cfg_path = Path(env.split('=',1)[1])
    cfg = json.loads(cfg_path.read_text())
    expected = SOURCE if arm == 'baseline' else TARGET
    assert cfg['args'][cfg['args'].index('--native')+1] == expected
    engines = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            if argv and argv[0] == str(STRATA/'build/strata') and '--serve' in argv:
                assert argv[argv.index('--native')+1] == expected
                engines.append({'pid':int(proc.name),'argv':argv,'proc_start_ticks':proc.joinpath('stat').read_text().split()[21]})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    assert len(engines) == 1, engines
    sys.path.insert(0,str(STRATA/'src/tools'))
    from gguf_reader import GGUFFile
    gguf = GGUFFile(expected)
    tensor = next(t for t in gguf.tensors if t.name == 'blk.45.ffn_down_shexp.weight')
    with open(expected,'rb') as stream:
        stream.seek(gguf.data_start+tensor.offset)
        tensor_sha = digest(stream.read(tensor.expected_bytes()))
    health = get('/health')
    assert health['loaded'] and health['status'] == 'ok'
    metrics = get('/metrics')
    return dict(at=now(), config_path=str(cfg_path), config=cfg, config_sha256=digest(cfg_path.read_bytes()),
                engines=engines, tensor_sha256=tensor_sha, model_size=Path(expected).stat().st_size,
                model_mtime_ns=Path(expected).stat().st_mtime_ns,
                health=health, router=get('/__reasoning_policy_health'), engine_metrics=metrics['engine'],live=metrics['live'])

def score(case, result):
    final = result['content']
    if case['kind'] == 'hle_mcq':
        sys.path.insert(0,str(PARENT))
        from hle_score import parse_answer
        predicted, raw = parse_answer('</think>'+final)
    else:
        boxes = re.findall(r'\\boxed\{([^{}]+)\}',final)
        raw = boxes[-1] if boxes else None
        predicted = raw.replace(',','').strip() if raw else None
    status = ('error' if result.get('error') else 'cutoff' if result['finish_reason']=='length' else
              'unparsed' if predicted is None else 'correct' if predicted==case['gold'] else 'wrong')
    text = result['reasoning_content']+'\n'+final
    words = text[-12000:].split()[-512:]
    grams = [tuple(words[i:i+8]) for i in range(max(0,len(words)-7))]
    duplicate = 1-len(set(grams))/len(grams) if grams else 0
    return dict(case=case['case'],id=case['id'],subject=case['subject'],seed=case['request']['seed'],
                gold=case['gold'],predicted=predicted,status=status,raw_answer=raw,
                completion_tokens=result.get('usage',{}).get('completion_tokens'),
                finish_reason=result['finish_reason'],wall_s=result['elapsed_s'],first_text_s=result.get('first_text_s'),
                duplicate8_tail=duplicate,wait_count=len(re.findall(r'\bwait\b',text,re.I)))

def run(arm):
    protocol_path = ROOT/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    outdir = ROOT/arm
    outdir.mkdir(exist_ok=False)
    start_identity = identity(arm)
    write(outdir/'identity_before.json',start_identity)
    rows = []
    for case in protocol['cases']:
        body = json.dumps(case['request'],sort_keys=True,separators=(',',':')).encode()
        assert digest(body) == case['request_sha256']
        print(json.dumps({'event':'request_start','arm':arm,'case':case['case'],'subject':case['subject'],'seed':case['request']['seed'],'time':now()}),flush=True)
        write(outdir/(case['case']+'.request.json'),case['request'])
        before_metrics = get('/metrics')
        t0 = time.monotonic()
        result = dict(case=case['case'],id=case['id'],request_sha256=case['request_sha256'],started_at=now(),
                      content='',reasoning_content='',finish_reason=None,usage={},timings={},live_before=before_metrics['live'])
        try:
            req = Request(protocol['endpoint'],data=body,headers={'Content-Type':'application/json'})
            with urlopen(req,timeout=1200) as response, (outdir/(case['case']+'.events.jsonl')).open('x') as events:
                result['http_status'] = response.status
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    elapsed = time.monotonic()-t0
                    if data == b'[DONE]':
                        result['done_received'] = True
                        break
                    event = json.loads(data)
                    events.write(json.dumps({'elapsed_s':elapsed,'event':event},ensure_ascii=False)+'\n')
                    events.flush()
                    if 'error' in event:
                        raise RuntimeError(str(event['error']))
                    result['response_id'] = event.get('id',result.get('response_id'))
                    for choice in event.get('choices',[]):
                        delta = choice.get('delta',{})
                        for key in ['content','reasoning_content']:
                            piece = delta.get(key) or ''
                            result[key] += piece
                            if piece and 'first_text_s' not in result:
                                result['first_text_s'] = elapsed
                        if choice.get('finish_reason'):
                            result['finish_reason'] = choice['finish_reason']
                    for key in ['usage','timings']:
                        if event.get(key):
                            result[key] = event[key]
            assert result['finish_reason'] is not None, 'Stream ended without finish reason'
            assert result['usage'].get('completion_tokens') is not None, 'Missing usage'
        except Exception as error:
            result['error'] = repr(error)
        result['elapsed_s'] = time.monotonic()-t0
        result['ended_at'] = now()
        write(outdir/(case['case']+'.result.json'),result)
        row = score(case,result)
        rows.append(row)
        write(outdir/'scores.json',dict(arm=arm,protocol_sha256=digest(protocol_path.read_bytes()),rows=rows,
                                      counts=dict(Counter(r['status'] for r in rows)),complete=len(rows)==len(protocol['cases'])))
        print(json.dumps({'event':'request_complete','arm':arm,**row}),flush=True)
        if result.get('error'):
            raise RuntimeError(result['error'])
    end_identity = identity(arm)
    write(outdir/'identity_after.json',end_identity)
    assert start_identity['engines'] == end_identity['engines'], 'Engine changed during this arm'
    assert start_identity['tensor_sha256'] == end_identity['tensor_sha256']
    print(json.dumps({'event':'arm_complete','arm':arm,'counts':dict(Counter(r['status'] for r in rows)),
                      'completion_tokens':sum(r['completion_tokens'] for r in rows)}),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['freeze','baseline','edited'])
    args=parser.parse_args()
    freeze() if args.operation=='freeze' else run(args.operation)
