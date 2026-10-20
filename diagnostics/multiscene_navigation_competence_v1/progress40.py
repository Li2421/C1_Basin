"""Queue and file-count progress only; never open evaluation outcomes."""
from pathlib import Path
from collections import Counter
import json
import re
import subprocess

HERE=Path(__file__).resolve().parent
queue=subprocess.check_output(['squeue','-h','-r','-o','%j|%T|%C|%b'],text=True)
rows=[line.split('|') for line in queue.splitlines()]
print(json.dumps(dict(cpu=sum(int(row[2]) for row in rows),gpu_shards=sum(sum(int(n) for n in re.findall(r'shard:(\d+)',row[3])) for row in rows),jobs=dict(Counter(row[0]+':'+row[1] for row in rows)),saved_evaluation_files={f:sum(1 for _ in (HERE/f'evaluation40_{f}'/'raw').glob('*.json')) for f in 'ACDE'},frozen_models=sum(1 for _ in (HERE/'fits40').glob('*/MODEL_FROZEN.receipt.json')),all_choices_frozen=(HERE/'ALL_AF_CHOICES40_FROZEN.receipt.json').exists(),analysis_frozen=(HERE/'AF_ANALYSIS_FROZEN.receipt.json').exists(),finalized=(HERE/'FINALIZATION40_COMPLETE.receipt.json').exists()),ensure_ascii=False))
