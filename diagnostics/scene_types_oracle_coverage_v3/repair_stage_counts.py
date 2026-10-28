"""Correct bookkeeping after recovery, without changing any numerical record."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main():
    changed=[]
    for p in HERE.glob('*_execution.json'):
        stage=p.name[:-len('_execution.json')];jobs=HERE/f'{stage}_jobs.json'
        if not jobs.exists():continue
        attempts=json.loads(jobs.read_text())
        if not attempts:continue
        data=json.loads(p.read_text());initial=attempts[0]['missing']
        if data.get('new')==initial and 'new_requested_at_first_submission'in data:continue
        backup=p.with_suffix('.before_count_fix.json')
        if not backup.exists():backup.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        data.update(new=initial,new_requested_at_first_submission=initial,
            new_unique_completed=max(0,initial-len(data.get('missing',[]))),
            accounting_fix='Recovery readback must not erase first-submission genuinely-missing counts. Raw traces, labels, UID and global actual-attempt budget unchanged.')
        p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');changed.append(stage)
    (HERE/'STAGE_ACCOUNTING_FIX.json').write_text(json.dumps(dict(stages=changed,numerical_records_changed=False,new_rollouts=0),ensure_ascii=False,indent=2)+'\n')
    print('Corrected stage accounting:',len(changed),'stages; no numerical labels changed.')

if __name__=='__main__':main()
