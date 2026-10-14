import json,subprocess,sys
from pathlib import Path
h=Path(__file__).resolve().parent
assert json.loads((h/'TRAINING_REPRODUCTION_GATE.json').read_text())['passed']
p=json.loads((h/'TRAINING_PLAN.json').read_text())[int(sys.argv[1])]
subprocess.run([sys.executable,'-B',str(h/'train_one.py'),p['run'],str(p['new_seed']),'new_flows/'+p['run']],check=True)
