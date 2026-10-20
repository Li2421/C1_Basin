from pathlib import Path
import json,hashlib,datetime
HERE=Path(__file__).resolve().parent;D=HERE.parent;out=[]
for n,receipt in [('ring_controller_critic_v1','MODEL_FREEZE.receipt.json'),('ring_critic_new_recipe_confirmation_v1','FINAL_FREEZE.receipt.json'),('ring_eta2_gap_v1','TEST_FREEZE.receipt.json')]:
 p=D/n;q=json.load(open(p/receipt));files=q['files'];paths=[f for f in (p/'predictions').glob('*.npz') if (n!='ring_controller_critic_v1' or any(f.name.startswith(prefix) for prefix in ('CV_fold','T1_','T2_','T3_')) and any(('_'+m+'_') in f.name for m in ('A0','A3')))]
 for f in paths:
  actual=hashlib.sha256(f.read_bytes()).hexdigest();expected=files.get(str(f.relative_to(p)));out.append(dict(path=str(f),receipt=str(p/receipt),hash=actual,expected=expected,passed=actual==expected))
for n in ['controller_new_checkpoint_confirmation_v1/fp32_n8','ring_independent_flow_confirmation_v2']:
 p=D/n;q=json.load(open(p/'SELECTOR_FROZEN.receipt.json'));actual=hashlib.sha256((p/'SELECTOR_MODELS.json').read_bytes()).hexdigest();out.append(dict(path=str(p/'SELECTOR_MODELS.json'),receipt=str(p/'SELECTOR_FROZEN.receipt.json'),hash=actual,expected=q['models_sha256'],passed=actual==q['models_sha256']))
assert all(r['passed'] for r in out);(HERE/'FROZEN_PREDICTION_AUDIT.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'files':out,'file_count':len(out)},indent=2)+'\n')
