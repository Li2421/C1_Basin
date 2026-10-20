"""Read-only history bootstrap; all output confined to cross-type5."""
from pathlib import Path
import hashlib,json,difflib,datetime
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'diagnostics/crosstype4_no_repair_schedule_diagnosis_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    names='core.py common.py registry.py resource_guard.py submit_aux.py mixed_execution.py mixed_runtime.py mixed_controllers.py generated_execution.py rollout_runtime.py new_execution.py unseen_models.py gradient_mpc.py conflict_based_search.py cbs_kernel.cpp cbs_kernel.so controllers.py zoo_runtime.py zoo_models.py ippo.py historical_controllers.py legacy_controllers.py learned.py gaussian.py right_hand.py trigger.py progress.py corrected_astar_reference.py fast_astar.py spacetime_astar.py astar_kernel.so probe_prefix.py history_audit.py legacy_conditioning.py frozen_networks.py finish_phase.py records.py early_reuse.py dynamic_reuse.py phase_complete.py catalog_names.py legacy_critic_model.py critic_model.py train_critic.py select_models.py inference.py'.split();rows=[]
    for name in names:
        source=OLD/name;dest=HERE/name;assert not dest.exists(),dest;data=source.read_bytes();new=data
        if source.suffix=='.py':
            s=data.decode()
            if name=='common.py':s=s.replace('exp_crosstype4_no_repair_schedule_diagnosis_v1','exp_crosstype5_repair_gate_late_eta_v1')
            if name=='registry.py':s=s.replace('Cross-type 4 no-repair and schedule diagnosis','Cross-type 5 repair gate and late eta proposals').replace("round='4'","round='5'")
            if name=='resource_guard.py':s=s.replace('18 CPU/2 GPU','24 CPU/5 GPU').replace('CPU_LIMIT=18;GPU_LIMIT=2','CPU_LIMIT=24;GPU_LIMIT=5').replace('crosstype4_','crosstype5_')
            if name=='submit_aux.py':s=s.replace('crosstype4_','crosstype5_')
            if name=='records.py':s=s.replace("ROOTS=[", "ROOTS=[D/'crosstype4_no_repair_schedule_diagnosis_v1',").replace("PHASES=('","PHASES=('A','B_training','B_evaluation','")
            new=s.encode()
        dest.write_bytes(new);rows.append(dict(source=str(source),source_sha256=sha(source),copy=str(dest),copy_sha256=sha(dest),diff=''.join(difflib.unified_diff(data.decode().splitlines(True),new.decode().splitlines(True))) if source.suffix=='.py' else 'byte-identical'))
    (HERE/'BOOTSTRAP_SOURCES.json').write_text(json.dumps(rows,indent=2)+'\n')
    for name in ('BASE_PROTOCOL.json','TRAINING_STATES.json','CONTROLLER_FEATURES.npz'):(HERE/name).write_bytes((OLD/name).read_bytes())
    (HERE/'PROMPT_OPENING_SNAPSHOT.md').write_bytes((ROOT/'docs/prompt_crosstype5_repair_gate_and_late_eta_proposals_20261019.md').read_bytes())
    (HERE/'RESOURCE_AUTHORIZATION.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu=24,gpu=5,user_text='现在你可以用24个cpu和五个gpu了',supersedes_prompt_resource_limits=True),indent=2)+'\n')
    print('Copied',len(rows),'implementations with source hashes;24CPU/5GPU authorized')
if __name__=='__main__':main()
