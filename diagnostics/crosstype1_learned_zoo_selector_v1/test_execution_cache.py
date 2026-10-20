"""No-physics replay of actions on saved probe states, including prefix restore."""
from common import *
from dataclasses import replace
from rollout_runtime import TracedObserved
from zoo_runtime import make
from probe_prefix import load_prefix,restore,clearance
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for


def main():
    p=read(HERE/'HISTORICAL_BASE_PROTOCOL.json');states=read(HERE/'PROBE_STATES.json')
    prefixes=read(HERE/'PROBE_PREFIX_INVENTORY.json')['prefixes']
    selected={}
    for row in prefixes:
        selected.setdefault(row['controller'],row)
    checks=[];cache={}
    for name,source in selected.items():
        base=TracedObserved(make(name,p,cache));env=base.env_cls(replace(base.config,terminate_on_collision=False,terminate_on_success=False))
        s=states[source['state_index']];physical=s['physical']
        env.reset(np.asarray(physical['positions']),velocities=np.asarray(physical['velocities']),goals=np.asarray(physical['goals']))
        c=dict(base_controller_name=name,arm=source['arm'],placement=source['placement'],seed=0)
        prefix=load_prefix(c,p,base,env,s)
        assert prefix is not None
        n=prefix['start_step'];assert env.step_count==n
        assert env.positions.tobytes()==np.asarray(prefix['positions'][-1]).tobytes()
        assert env.velocities.tobytes()==np.asarray(prefix['velocities'][-1]).tobytes()
        eta=np.asarray(p['eta'][10 if c['arm']=='F10' else 13],np.float64)
        matches=[]
        for t in sorted(set([0,min(1,n-1),min(100,n-1),n-1])) if n else []:
            restore(env,prefix['positions'][t],prefix['velocities'][t],t)
            action,_,_=base.action(env,key_for(s,0,t),eta,'field' if c['placement']=='FF' else 'old')
            matches.append(np.asarray(action).tobytes()==np.asarray(prefix['actions'][t]).tobytes())
            assert np.isfinite(prefix['swept'][t])
        checks.append(dict(controller=name,stored_steps=n,checked_actions=len(matches),action_bytes_equal=all(matches),
                           failed_prefix_preserved=not prefix['can_continue'],source_raw=source['raw_path']))
        print(name,'saved-state action equality',all(matches),flush=True)
    # The schedule uses the old convention: action indices < t use eta13.
    for switch in (100,300,500):
        assert [13 if t<switch else 10 for t in (0,switch-1,switch,699)]==[13,13,10,10]
    write(HERE/'EXECUTION_CACHE_TESTS.json',dict(utc=utc(),passed=all(r['action_bytes_equal'] for r in checks),
          controllers=len(checks),checks=checks,physical_steps=0,new_rollouts=0,
          prefix_failure_not_reexecuted=True,source_checkpoint_only_inference=True))


if __name__=='__main__':main()
