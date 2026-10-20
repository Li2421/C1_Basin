"""Resume after analysis-only path concatenation correction; never resubmit A."""
from core import *
from orchestrate import call,submit_wait,freeze_rhr
from audit_utils import source_inventory
import traceback

def main():
    assert read(HERE/'A_MERGE_RECEIPT.json')['records']==1280;assert (HERE/'A_RESULTS.json').exists();assert not (HERE/'tune_PREFLIGHT.receipt.json').exists()
    write(HERE/'ANALYSIS_RECOVERY.json',dict(utc=utc(),previous_status=read(HERE/'RUN_STATUS.json'),reason='Path/string concatenation TypeError in analysis output path only; fixed analysis.py. No rollout/controller/RNG semantics changed; A not resubmitted.'))
    seal('ANALYSIS_READY.receipt.json',[HERE/'analysis.py',HERE/'finish.py'],before_tune_and_K16_labels=True)
    for phase in ['tune','fair','C']:
        write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase=phase,status='DESIGN'));call('planning.py',phase);(HERE/'main_protocol.json').write_bytes((HERE/f'{phase}_protocol.json').read_bytes());submit_wait(phase);call('postflight.py',phase)
        if phase=='tune':freeze_rhr()
        if phase=='fair':call('analysis.py','B')
    call('analysis.py','C');call('finish.py');write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase='delivery',status='COMPLETE'));source_inventory('imports_resume.json')
if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc();write(HERE/'RUN_STATUS.json',dict(utc=utc(),status='STOPPED',error=error));raise
