"""Supplement coverage honestly; never invent an opening whole-file hash."""
from experiment import *
import csv

def main():
    source=ROOT/'diagnostics/crosstype3_unseen_controller_schedule_v1/ZOO_MANIFEST.csv';kinds={r['type_code'] for r in read(HERE/'CONTROLLER_MANIFEST.json')}
    selected=[r for r in csv.DictReader(source.open()) if r['type_code'] in kinds]
    inv=ROOT/'diagnostics/axis3_cross_scene_v1/HISTORICAL_INVENTORY.json'
    assert selected==read(HERE/'CONTROLLER_MANIFEST.json');assert read(inv)['eta']==read(HERE/'FIXED_ETA.json')
    entries=[dict(path=str(source),whole_file_sha256=sha(source),opening_whole_file_hash=None,opening_used_view=str(HERE/'CONTROLLER_MANIFEST.json'),used_view_matches=True),dict(path=str(inv),whole_file_sha256=sha(inv),opening_whole_file_hash=None,opening_used_view=str(HERE/'FIXED_ETA.json'),used_view_matches=True)]
    f=read(HERE/'FROZEN_PIPELINE_INSPECTION.json');entries.append(dict(path=str(ROOT/f['path']),whole_file_sha256=sha(ROOT/f['path']),opening_whole_file_hash=f['sha256']))
    f=read(HERE/'LEARNING_MENU.receipt.json');entries.append(dict(path=f['source'],whole_file_sha256=sha(f['source']),opening_whole_file_hash=f['source_sha256']))
    for r in read(HERE/'KERNEL_PORTING.json'):entries.append(dict(path=r['source'],whole_file_sha256=sha(r['source']),opening_whole_file_hash=r['source_sha256']))
    for path,h in read(HERE/'PILOT_REUSED_SOURCE_START.json')['files'].items():entries.append(dict(path=path,whole_file_sha256=sha(path),opening_whole_file_hash=h,opening_phase='before pilot execution'))
    assert all(e['opening_whole_file_hash'] is None or e['opening_whole_file_hash']==e['whole_file_sha256'] for e in entries)
    write(HERE/'SOURCE_INPUT_AUDIT_SUPPLEMENT.json',dict(utc=utc(),entries=entries,coverage_limitation='ZOO_MANIFEST.csv与HISTORICAL_INVENTORY.json未在开工时记录整个原文件sha256；实际复用的48行controller参数及eta字段均在首条rollout前封存，当前逐值一致。补录whole-file hash不能替代缺少的开工全文件收据。',strict_opening_full_file_coverage=False))
    seal('SOURCE_INPUT_AUDIT_SUPPLEMENT.receipt.json',[HERE/'SOURCE_INPUT_AUDIT_SUPPLEMENT.json',HERE/'source_input_audit.py'],actual_used_views_match_opening_sealed_copies=True,opening_whole_file_gap_explicit=True)

if __name__=='__main__':main()
