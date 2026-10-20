from core import *
from fair_rhr import Trigger
from analysis import collect,historical,paired
from audit_utils import source_inventory
from types import SimpleNamespace
import jax
jax.config.update('jax_enable_x64',False)
def main():
    if (HERE/'RHR_TRACE_AUDIT.json').exists():return
    groups={};checked=0;differences=[];mainG={}
    for phase in ['tune','fair']:
        db={r['rollout_uid']:r for r in read(HERE/f'{phase}_DB_ROWS.json')}
        for c in read(HERE/f'{phase}_CASES.json'):
            raw=read(HERE/phase/'raw'/f"{c['rollout_uid']}.json")
            with np.load(raw['trace_path']) as z:
                pos=z['positions'];vel=z['velocities'];goals=z['goals'];actual=z['rhr_active'];meta=json.loads(str(z['metadata_json']));cfg=SimpleNamespace(goal_tolerance=meta['config']['goal_tolerance']);tr=Trigger(c['rhr_rule']);expected=[]
                for t in range(len(pos)-1):expected.append(tr.update(SimpleNamespace(step_count=t,positions=pos[t],velocities=vel[t],goals=goals,config=cfg)))
                if not np.array_equal(np.asarray(expected,bool).reshape(-1,4),actual):differences.append(c['rollout_uid'])
                if phase=='fair' and c['kind']=='G':mainG[(c['state_index'],c['seed'])]=dict(positions=pos.copy(),actions=z['actions'].copy(),velocities=vel.copy(),success=raw['success'])
                k=(phase,'students' if c['kind']=='student' else 'G',c['arm']);g=groups.setdefault(k,dict(episodes=0,steps=0,active_agent_steps=0,active_episodes=0));g['episodes']+=1;g['steps']+=len(actual);g['active_agent_steps']+=int(actual.sum());g['active_episodes']+=int(actual.any());checked+=1
    assert not differences,differences
    deterministic=all(all(np.array_equal(mainG[(s,0)][k],mainG[(s,1)][k]) for k in ['positions','velocities','actions','success']) for s in range(32));assert deterministic
    write(HERE/'RHR_TRACE_AUDIT.json',dict(utc=utc(),passed=True,episodes=checked,rule_recomputation_mismatches=differences,G_main_seed_exact=deterministic,G_main_independent_states=32,G_main_rollouts=64,groups=[dict(phase=k[0],pool=k[1],arm=k[2],**g,active_agent_step_fraction=g['active_agent_steps']/max(4*g['steps'],1)) for k,g in groups.items()],jax_enable_x64=bool(jax.config.x64_enabled)))
    r=read(HERE/'RHR_GRID_RESULTS.json');lines=['| Pool | Mode | Window(s) | Speed(m/s) | Tuning success | Active episodes | Agent-step activation |','|---|---|---:|---:|---:|---:|---:|']
    for pool in ['students','G']:
        for q in r['pools'][pool]:
            rule=q['rule'];g=groups[('tune',pool,f"RHR_grid{q['index']:02}")];lines.append(f"| {pool} | {rule['mode']} | {rule['steps']*.05:g} | {rule['speed']:.2f} | {q['lower']:.2%} | {g['active_episodes']}/{g['episodes']} | {g['active_agent_steps']/max(4*g['steps'],1):.2%} |")
    (HERE/'RHR_GRID_TABLE.md').write_text('\n'.join(lines)+'\n')
    a=read(HERE/'A_RESULTS.json');lines=['| 管线 | Controller | safety | F10 | F13 | S300 | S500 | S500−学习型参考 [95%CI, pp] |','|---|---|---:|---:|---:|---:|---:|---|']
    for pipe in ['local','central']:
        for n in ['Z','G']:
            q=a[pipe]['controls'][n]['arms'];d=q['S500']['minus_reference'];ci=d['CI_unknown_envelope'];lines.append('| '+pipe+' | '+n+' | '+' | '.join(f"{q[k]['mean']:.2%}" for k in ['safety baseline','F10','F13','S300','S500'])+f" | {100*d['mean']:.2f} [{100*ci[0]:.2f}, {100*ci[1]:.2f}] |")
    (HERE/'A_RESULTS_TABLE.md').write_text('\n'.join(lines)+'\n')
    sy,su=historical(False);gy,gu=collect('fair',['G'],['RHR-fair']);classic=paired(sy,su,np.broadcast_to(gy,sy.shape),np.broadcast_to(gu,su.shape))[0]['minus_reference'];write(HERE/'CLASSICAL_COMPARISON.json',dict(utc=utc(),comparison='students+S500 minus G+CBF+RHR-fair; descriptive different-controller-family comparison, not the preregistered same-student arm contrast',delta=classic,interpretation='Classic G+RHR succeeds on every tested state. Same-student S500 superiority does not establish superiority over the classic G+RHR controller.',independent_states=32))
    source_inventory('imports_delivery_tables.json')
if __name__=='__main__':main()
