"""Read-only evidence audit; recompute physical label counts, provenance and gates."""
from pathlib import Path
import json,hashlib,datetime,collections
import numpy as np
from .main_analysis import write
HERE=Path(__file__).resolve().parent

def main():
 src=HERE.parent/'controller_ff_factorial_repair_v1';z=np.load(src/'dataset.npz');r=np.load(src/'raw_labels.npz');es=[0,5,10,15];ix=np.char.startswith(z['controller'],'flow_');o=r['observed'][ix][:,es];u=r['unknown'][ix][:,es]
 sources={}
 for name in ['controller_geometry_identifiability_v1','controller_state_transfer_v1','controller_ff_factorial_repair_v1']:
  path=HERE.parent/name;sources[name]={'path':str(path.resolve()),'compatibility':{'controller_geometry_identifiability_v1':'DB goal-reference FF,C5,K4; different mechanism/Flow reference from Toy. Reuse historical evidence, do not pool labels blindly.','controller_state_transfer_v1':'EXACT execution-compatible Toy width/Flow/operator labels; primary reuse, exact identity checked.','controller_ff_factorial_repair_v1':'compatible baseline C5 Flow K4 across four scenes; reused saved LOSO predictions and observed seeds for disjoint calibration.'}[name], 'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in path.iterdir() if f.is_file() and f.name in ['protocol.json','protocol_main.json','raw_labels.npz','dataset.npz','results.json','entity_results.json','reproduction.json','PREREGISTRATION.md']}}
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':sources,'legacy_factorial_flow_tensor_capacity':int(o.size),'actual_observed_flow_rollouts':int(o.sum()),'unknown_flow_rollouts':int(u.sum()),'legacy_flow_collisions':int(r['collision'][ix][:,es][o>0].sum()),'historical_count_correction':'The audit prose mentions12800 Flow outcomes; this is tensor capacity. raw_labels.observed identifies7680 actual outcomes, including51 unknown. Never treat unobserved cells as failures.','new_training_necessity':'No saved model uses this experiment training-state8..23 split and exact MANY/FEW equal budgets. Existing LOSO calibration predictions are reused without retraining.','new_rollout_necessity':'two isolated developmental pilots; precision supplement1024missing exact seeds; four horizon reproduction probes; new crossed topology/N/navigation profiles3072. Changing critic features causes zero new rollouts.'}
 write(HERE/'history_audit.json',out);print({k:v for k,v in out.items() if k!='sources'})
if __name__=='__main__':main()
