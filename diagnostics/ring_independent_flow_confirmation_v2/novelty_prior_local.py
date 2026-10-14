from core import *
import pickle
from diagnostics.controller_new_checkpoint_confirmation_v1.fp32_n8.compatibility import param_leaves
def ph(p):
 x=pickle.loads(p.read_bytes());h=hashlib.sha256()
 for path,a in param_leaves(x['agent']['network']['params']).items():h.update(path.encode());h.update(str(a.dtype).encode());h.update(json.dumps(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
prior=[p for parent in ['reproduction_cpu','reproduction_gpu','backup','new_flows/base_u_plus_w_v4_wide_macflow'] for p in (HERE/parent).glob('**/*.pkl')];rows=[dict(path=str(p),file_sha256=sha(p),param_sha256=ph(p)) for p in prior];targets=read(HERE/'CHECKPOINT_NOVELTY_AUDIT.json')['checkpoints'];assert all(x['new_param_sha256'] not in {r['param_sha256'] for r in rows} for x in targets);write(HERE/'PRIOR_LOCAL_CHECKPOINT_NOVELTY_AUDIT.json',dict(utc=utc(),passed=True,prior_checkpoints=rows,checked_targets=14,new_target_success_labels_read=False,scope='Additional prior v2 reproduction/backup and invalid wide artifacts; training outputs of each current target run are not prior independent controllers'))
print('Novelty against local prior reproduction/backup/invalid checkpoints PASS')
