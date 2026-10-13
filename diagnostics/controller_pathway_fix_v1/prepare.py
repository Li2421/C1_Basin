from util import *
import subprocess
def main():
    tasks=[]
    for t in task_list('blend'):
        for arm in ('B0','B1','B2'):tasks.append(dict(t,arm=arm))
    for t in task_list('real'):
        for arm in ('B0','B1'):tasks.append(dict(t,arm=arm))
    write(HERE/'REPLAY_TASKS.json',tasks)
    fits=[dict(t,arm=a) for t in task_list('blend') for a in NEW_ARMS]
    write(HERE/'FIT_TASKS_BLEND.json',fits)
    files=set([ROOT/'AGENTS.md',ROOT/'docs/prompt_round6_controller_pathway_fix_20261013.md'])
    for root in (R4,R5):
        for pattern in ('*.py','*REPORT.md','*HANDOFF.md','*results.json','*protocol.json','*LABELS.npz','*GATE_RESULTS.json'):
            files.update(root.glob(pattern))
        for sub in ('stage_b','stage_2','fits','static_features','stage_b/static_features','stage_2/static_features','stage_2/fits','stage_2/shared_sparse_fits'):
            q=root/sub
            for pattern in ('*.py','*.json','*.npz','*/*.msgpack','*/*.npz','*/complete.json','*/MODEL_FROZEN.json'):files.update(q.glob(pattern))
    val=ROOT/'diagnostics/controller_q_transfer_round3_v1'
    files.update([val/'P1_LABELS.npz',val/'P1_EXECUTION_PROTOCOL.json',val/'state_generator.py'])
    files.update((val/'p1_features').glob('*.npz'))
    index=[dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(files) if f.is_file()]
    write(HERE/'HISTORICAL_ASSET_INDEX.json',dict(utc=utc(),files=index,compatibility='Existing exact execution labels reused; only prediction method changes',missing_evidence='No checkpoints for prescribed PCA/residual/logistic arms; F3 requires no new fit',round4_stage_b_and_c='Read background; Stage2 testbed uses Stage A seven-fold labels and checkpoints'))
    from shared_rollout_db.src.rollout_db import DB_PATH
    write(HERE/'DB_READ_ONLY_BASELINE.json',dict(path=str(DB_PATH),sha256=sha(DB_PATH),bytes=DB_PATH.stat().st_size,mtime_ns=DB_PATH.stat().st_mtime_ns,connection_mode='No database connection needed by new fitting/analysis scripts'))
    print(json.dumps(dict(replay_tasks=len(tasks),blend_fit_prediction_tasks=len(fits),sources_hashed=len(index))))
if __name__=='__main__':main()
