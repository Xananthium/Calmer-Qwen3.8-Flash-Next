#!/usr/bin/env python3
"""Descriptive whole-output repetition metrics; no inferred semantic labels."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import re

ROOT=Path('/home/researcher/model-tools/qwen38-weight-edit-20261104/strata_l45plus_nearby_natural_20261118')

def measure(result):
    text=result['reasoning_content']+'\n'+result['content']
    words=text.split()
    result_metrics={'generated_words':len(words),'wait_count':len(re.findall(r'\bwait\b',text,re.I))}
    for size in [8,20]:
        counts=Counter(tuple(words[i:i+size]) for i in range(max(0,len(words)-size+1)))
        total=sum(counts.values())
        result_metrics[f'duplicate{size}_full']=1-len(counts)/total if total else 0
        result_metrics[f'max_{size}gram_occurrences']=max(counts.values(),default=0)
    paragraphs=[' '.join(p.split()) for p in re.split(r'\n\s*\n',text) if len(p.split())>=12]
    counts=Counter(paragraphs)
    result_metrics['exact_repeated_paragraphs']=sum(n-1 for n in counts.values() if n>1)
    return result_metrics

def main():
    rows=[]
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for case in protocol['cases']:
        row={'case':case['case'],'subject':case['subject']}
        for arm in ['baseline','edited']:
            p=ROOT/arm/(case['case']+'.result.json')
            if p.exists():row[arm]=measure(json.loads(p.read_text()))
        rows.append(row)
    report=dict(at=datetime.now(timezone.utc).isoformat(),rows=rows,
        scope='Descriptive supplemental analysis over emitted output only, excluding prompt repetitions. Added after baseline, before long edited-panel results.',
        limitations=['Repeated mathematical notation and valid rechecking can raise exact phrase metrics.','This does not classify reasoning usefulness or diagnose a looping mechanism.'])
    (ROOT/'whole_output_repetition.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
