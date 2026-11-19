#!/usr/bin/env python3
"""Verify published file identities and replay the main-panel token totals without a GPU."""
import hashlib,json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
archive=json.loads((root/'evidence/archive_manifest.json').read_text())
checked=0
for row in archive['files']:
    if 'public_path' not in row: continue
    p=root/row['public_path']; b=p.read_bytes()
    assert len(b)==row['public_bytes'],p
    assert hashlib.sha256(b).hexdigest()==row['public_sha256'],p
    if p.suffix=='.json': json.loads(b)
    checked+=1
comparison=json.loads((root/'evidence/comparison.json').read_text())
for label,group in comparison['common_case_variants'].items():
    total=sum(row['completion_tokens'] for row in group['rows'])
    assert total==group['completion_tokens'],label
    print(f'{label}: {total:,} tokens'+(' (lower bound)' if group['completion_tokens_are_lower_bound'] else ''))
m=json.loads((root/'release/release_manifest.json').read_text())
for edit in m['edits']:
    b=(root/'release'/edit['patch']).read_bytes()
    assert len(b)==edit['bytes'] and hashlib.sha256(b).hexdigest()==edit['edited_sha256']
print(f'Passed: {checked} archived file identities, comparison totals, selected tensor payloads.')
