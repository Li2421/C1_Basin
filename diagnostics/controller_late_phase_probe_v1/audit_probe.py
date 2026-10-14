from core import *
from types import SimpleNamespace
from probe import stats

def main():
    prereg();p=read(HERE/'probe_protocol.json');cases=read(HERE/'probe_CASES.json');complete=0;maximum_error=0.;steps=0
    for c in cases:
        raw=read(HERE/'probe/raw'/f"{c['probe_uid']}.json");assert raw['case']==c;assert not any(k in raw for k in ['success','outcome','timeout','collision','summary']);assert raw['terminal_labels_read'] is False;assert sha(Path(raw['trace_path']))==raw['trace_sha256'];z=np.load(raw['trace_path']);prefix=z['prefix_positions'];assert len(prefix)==raw['prefix_steps']+1;steps+=raw['prefix_steps']
        for bi,b in enumerate(raw['branches']):
            path=z[f'branch{bi}_positions'];actions=z[f'branch{bi}_actions'];assert np.array_equal(path[0],prefix[-1]);assert len(actions)==b['steps'] and len(path)==b['steps']+1;steps+=b['steps']
            if b['complete']:
                env=SimpleNamespace(goals=z['goals'],config=SimpleNamespace(goal_tolerance=.09));v,zero=stats(env,path[0],path,actions,z[f'branch{bi}_projection_active']);err=float(np.max(abs(np.asarray(v)-b['statistics'])));assert err<1e-12;maximum_error=max(maximum_error,err);assert zero==b['zero_progress_denominator'];assert raw['prefix_steps']==400 and b['steps']==200;complete+=1
            else:assert b['statistics'] is None and b['error'] is not None
    for item in read(HERE/'HISTORICAL_ASSET_INDEX.json')['assets']:assert sha(ROOT/item['path'])==item['sha256']
    write(HERE/'PROBE_REPRODUCTION_AUDIT.json',dict(status='PASS',requests=len(cases),known_branches=complete,statistics_max_abs_error=maximum_error,all_branch_initial_positions_exact=True,actual_physical_steps=steps,historical_inputs_unchanged=True,terminal_labels_read=False))
    print(json.dumps(dict(status='PASS',requests=len(cases),known_branches=complete,steps=steps)))
if __name__=='__main__':main()
