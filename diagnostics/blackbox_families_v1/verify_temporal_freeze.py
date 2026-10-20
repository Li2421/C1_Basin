"""Verify choice freeze precedes every main submission and main raw artifact."""
from common import *
from datetime import datetime
freeze=read(HERE/'ALL_CHOICES_FROZEN.receipt.json');t=datetime.fromisoformat(freeze['utc']).timestamp();submitted=read(HERE/'main_SUBMISSION.json');assert datetime.fromisoformat(submitted['utc']).timestamp()>t
raw=list((HERE/'main/raw').glob('*.json'));earliest=min((p.stat().st_mtime for p in raw),default=None);assert earliest is None or earliest>t
for path,digest in freeze['files'].items():assert sha(path)==digest,path
write(HERE/'TEMPORAL_FREEZE_VERIFICATION.json',dict(utc=utc(),passed=True,freeze_utc=freeze['utc'],main_submission_utc=submitted['utc'],earliest_main_raw_mtime_utc=datetime.fromtimestamp(earliest,timezone.utc).isoformat() if earliest else None,raw_count=len(raw),selection_files_unchanged=True))
print('all choice receipts precede main labels; sealed selections unchanged')
