"""Recompute the v1 weak-witness diagnosis from saved simulator evidence."""
from core import *

def main():
    old=ROOT/'diagnostics/scene_types_basis_dimension_v1';requests={}
    for p in old.glob('calibration_*_planned.json'):
        for c in read(p)['exact_configs']:
            if c['purpose']=='joint_feasibility_witness':requests[c['rollout_uid']]=c
    result=[]
    for uid,c in requests.items():
        p=old/'raw'/f'{uid}.json'
        if not p.exists():continue
        r=read(p)
        with np.load(r['trace'],allow_pickle=False)as z:
            positions=z['positions'];vel=z['velocities'];dt=c['scenario_payload']['config']['dt'];window=min(len(positions)-1,round(2/dt))
            never=np.max(np.linalg.norm(positions-positions[0],axis=-1),axis=0)<1e-9
            last=np.linalg.norm(positions[-1]-positions[-1-window],axis=-1)if window else np.full(len(never),np.nan)
            still=bool(window and np.max(last)<1e-6 and np.max(np.linalg.norm(vel[-window:],axis=-1))<1e-6)
            result.append(dict(uid=uid,type_id=c['type_id'],level=c['scenario_payload'].get('factor'),scene=c['scene'],state_index=c['state_index'],success=r['success'],outcome=r['outcome'],agents=len(never),never_moved=int(never.sum()),never_moved_agent_ids=np.flatnonzero(never),last2s_all_still=still,available_last_window_seconds=window*dt,original_parameters=c['controller_payload']['nominal']['parameters']))
    files=[old/'generated/backend'/n for n in ('tiled_planning_pa_memo_query_disk_candidate_spatial_mask_20261022.py','reservation_certificate_pa_memo_query_disk_candidate_spatial_mask_20261022.py')]
    dump(HERE/'WITNESS_DIAGNOSTICS.json',dict(cases=result,n=len(result),success=sum(r['success']for r in result),with_never_moved=sum(r['never_moved']>0 for r in result),last2s_all_still=sum(r['last2s_all_still']for r in result),sources={str(p):sha(p)for p in files},interpretation='Original second parameter100000 is the replan interval, so only initial planning occurred. Priority planning is incomplete and path=None reserves the stationary start. Saved stationary trajectories diagnose witness-tool weakness, not joint impossibility or deadlock.'))
    print('Read-only witness diagnostics',len(result),'records',flush=True)
if __name__=='__main__':main()
