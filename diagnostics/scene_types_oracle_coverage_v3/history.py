"""Index every v1 phase by physical execution semantics, not by human labels."""
from core import *
from collections import Counter
from functools import lru_cache

OLD=ROOT/'diagnostics/scene_types_basis_dimension_v1'

def key_parts(sc,physical,nominal,spec,seed=0):
    config=dict(sc['config']);config.pop('max_steps',None)
    return digest(dict(config=config,walls=sc['walls'],bottleneck_config=sc.get('bottleneck_config'),
                       route_guidance=bool(sc['route_guidance']),physical=physical,
                       nominal=nominal,dictionary=spec,seed=seed,safety='distributed',precision='jax_x64_false'))

def request_key(c):
    return key_parts(c['scenario_payload'],{k:c['state'][k]for k in ('positions','velocities','goals')},
                     c['controller_payload']['nominal'],c['spec'],c['seed'])

@lru_cache(maxsize=1)
def index():
    p=HERE/'HISTORY_INDEX.json'
    if p.exists():return read(p)['records']
    source_hashes={str(OLD/n):sha(OLD/n)for n in ('runtime.py','routes.py')}
    sources={};scenarios={};records={};rejected=Counter();counts=Counter()
    with db.connect(True)as con:
        for path in sorted((OLD/'raw').glob('*.json')):
            r=read(path)
            if r['controller']not in ('centralized_sampling','gradient_mpc'):continue
            cid=r['controller_uid'];sid=r['scenario_uid']
            if cid not in sources:
                row=con.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(cid,)).fetchone()
                if not row:rejected['controller_config_missing']+=1;continue
                sources[cid]=json.loads(row[0])
            ctl=sources[cid]
            if 'nominal'not in ctl or 'dictionary'not in ctl:continue
            execution=ctl.get('execution_sources',{})
            if any(execution.get(path)!=value for path,value in source_hashes.items()):
                rejected['different_v1_runtime_or_route_revision']+=1;continue
            current_shared={str(ROOT/'shared_control'/n):sha(ROOT/'shared_control'/n)for n in ('basis_dictionary.py','hard_projection.py','basis_families.py','correction_b5.py')}
            if any(execution.get(path)!=value for path,value in current_shared.items()):
                rejected['different_shared_control_revision']+=1;continue
            if sid not in scenarios:
                row=con.execute('SELECT metadata_json FROM scenario WHERE scenario_uid=?',(sid,)).fetchone()
                if not row:rejected['scenario_metadata_missing']+=1;continue
                scenarios[sid]=json.loads(row[0])
            sc=scenarios[sid]
            if sc.get('version')!='scene_types_basis_dimension_v1':continue
            physical=dict(positions=r['initial_positions'],velocities=r['initial_velocities'],goals=r['goals'])
            key=key_parts(sc,physical,ctl['nominal'],ctl['dictionary'],r['seed'])
            item=dict(rollout_uid=r['rollout_uid'],path=str(path),trace=r['trace'],success=r['success'],
                      collision=r['collision'],timeout=r['timeout'],numerical_failure=r['numerical_failure'],
                      outcome=r['outcome'],controller=ctl['nominal'],spec=ctl['dictionary'],
                      scenario=sc,type_id=r['type_id'],state_index=r['state_index'],solo=r['solo'],purpose=r['purpose'])
            if key in records:
                assert all(records[key][v]==item[v]for v in ('success','collision','timeout','numerical_failure','outcome')),('HISTORICAL_CONFLICT',key)
            else:records[key]=item
            counts[r['purpose']]+=1
    dump(p,dict(records=records,accepted_by_purpose=dict(counts),excluded=dict(rejected),
                rule='All v1 raw phases considered; only final literal v1 runtime/routes/shared-control revisions and identical physical requests admitted. Horizon excluded only for explicit prefix-certified reuse.'))
    print('HISTORY',len(records),'compatible requests;',dict(rejected),flush=True)
    return records

def load_for(c):return index().get(request_key(c))

if __name__=='__main__':index()
