"""Reuse unchanged exact records; register only native-cadence evidence gaps."""
from native_io import *
from clock_plant import PERIODS
import csv,collections,copy
OLD=ROOT/'diagnostics/c1_v3_new_baseline_reproduction_20261103'
PAIRED=ROOT/'diagnostics/c1_v3_mcbf_circulation_paired_20261104'
AUDIT=ROOT/'diagnostics/external_method_time_discretization_audit_20261104'
ROOTS=[ROOT/'diagnostics'/n for n in ['remaining_evaluations_v1','b_family_external_paper_scenes_v1','b_family_final_locked_test_v1','b_family_final_cross_controller_v1']]

def raw_path(uid):
    paths=[p/'raw'/(uid+'.json') for p in [OLD,PAIRED,*ROOTS]]
    found=next((p for p in paths if p.exists()),None)
    assert found,('missing historical record',uid)
    return found

def main():
    old=read(OLD/'ACTIVE_COMPARISON_CASES.json');physical={c['payload']['physical_key']:c for c in old}
    rows=[];seen=set()
    with (AUDIT/'EVALUATED_RATE_INVENTORY.csv').open(encoding='utf-8-sig') as f:inventory=list(csv.DictReader(f))
    for r in inventory:
        if r['method']=='chen':continue
        key,method,uid=r['physical_key'],r['method'],r['rollout_uid']
        if (key,method,uid) in seen:continue
        seen.add((key,method,uid));path=raw_path(uid);source=read(path)
        if key in physical:prior=physical[key]
        else:prior=source['case']
        scenario=copy.deepcopy(prior['scenario']);state=copy.deepcopy(prior['state'])
        assert state['positions']==source['initial_positions']
        assert state['goals']==source['goals']
        period=PERIODS[method];same=scenario['config']['dt']==period
        row=dict(physical_key=key,method=method,cohort=r['cohort'],old_uid=uid,old_raw=str(path),old_raw_sha256=sha(path),
            old_trace=source.get('trace'),old_trace_sha256=source.get('trace_sha256'),old_period_s=scenario['config']['dt'],
            control_period_s=period,score_period_s=scenario['config']['dt'],reuse=same,reason='unchanged controller, cadence, SI propagation, scoring and horizon' if same else 'native controller period differs; new control trajectory necessary')
        if same:
            row.update(native_uid=uid,native_raw=str(path));rows.append(row);continue
        payload=dict(method=method,physical_key=key,cohort=r['cohort'],control_period_s=period,
            score_period_s=scenario['config']['dt'],time_profile='native_callback_ZOH_original_score_clock_v1',
            comparison='existing complete-system common-SI port with author nominal and native callback rate',
            changes=['controller cadence only; preserve author inner time scales, counters, solver and options','score only on original scenario clock; same physical horizon'],
            external_root=str(PAIRED) if scenario.get('external_motion') else None,
            prior_uid=uid,new_training=False,test_tuning=False)
        if method in ['dac','mgr']:
            cp=source['case']['controller_payload'];adapter=Path(cp['paper_method_adapter']);root=adapter.parent
            payload['native_static_geometry']=source['case']['state'].get('native_static_geometry')
            extension=False
            if adapter.name=='eps1_paper_adapter.py':
                native=payload['native_static_geometry'] or {};bounds=native.get('physical_bounds');options=source['paper_options']
                compatible=method=='dac' or (not options.get('_author_circles_in_common_chart') and (bounds is None or bounds[0]==bounds[1]==0))
                adapter=root/('eps1_legacy_paper_adapter.py' if compatible else 'bfc_paper_adapter.py');extension=not compatible and method=='mgr'
            payload.update(legacy_root=str(root),legacy_adapter=str(adapter),open_circle_extension=extension,
                author_options=source['paper_options'],author_seed=source['seed'])
        row.update(_scenario=scenario,_state=state,_payload=payload);rows.append(row)
    # Freeze the complete numerical source closure used by these private workers.
    paths={D/n for n in ['native_io.py','clock_plant.py','adapters.py','worker.py']}
    paths.update([OLD/'common_plant.py',OLD/'plant_snapshot/environment.py',OLD/'plant_snapshot/swap_geometry.py',
        OLD/'circulation_port_v2.py',OLD/'mcbf_port_v4.py',OLD/'capsule_lidar.py',PAIRED/'author_adapters.py'])
    for root in [OLD/'vendor',*[p/'inputs/diagnostics/paper_baselines_20261028/vendor/dac' for p in ROOTS],
        *[p/'inputs/diagnostics/paper_baselines_20261028/vendor/mgr' for p in ROOTS],
        *[p/'inputs/diagnostics/paper_baselines_20261028/vendor/qpswift/python' for p in ROOTS],
        *[p/'inputs/diagnostics/paper_baselines_20261028/deps' for p in ROOTS]]:
        if root.exists():paths.update(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix in ['.py','.so','.yaml','.yml'])
    for row in rows:
        if '_payload' not in row:continue
        payload=row['_payload']
        if payload['method'] in ['dac','mgr']:
            root=Path(payload['legacy_root']);paths.update([Path(payload['legacy_adapter']),root/'ports/paper_adapters.py',root/'paper_methods/dac_author_si/adapter.py'])
            if payload['open_circle_extension']:paths.add(root/'eps1_open_circle_paper_adapter.py')
    profile={str(p.resolve()):sha(p) for p in sorted(paths)};dump(D/'EXECUTION_SOURCE_PROFILE.json',profile)
    cases=[]
    for row in rows:
        if row['reuse']:continue
        payload=row.pop('_payload');payload['execution_profile']=str(D/'EXECUTION_SOURCE_PROFILE.json');payload['execution_profile_sha256']=sha(D/'EXECUTION_SOURCE_PROFILE.json')
        payload['python']=str(OLD/'.venv/bin/python') if payload['method'] in ['circulation','mcbf'] else '/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
        c=make_case(row['method']+'_'+row['physical_key']+'_native',row.pop('_scenario'),row.pop('_state'),payload)
        cases.append(c);row.update(native_uid=c['rollout_uid'],native_raw=str(D/'raw'/(c['rollout_uid']+'.json')))
    assert len({c['rollout_uid'] for c in cases})==len(cases)
    dump(D/'HISTORY_CHECK.json',dict(previous_audit=str(AUDIT/'REPORT.txt'),prior_profiles=[str(OLD/'FINAL_DELIVERY_CHECK.json'),str(PAIRED/'FINAL_DELIVERY_CHECK.json')],
        scope='all existing fixed external method comparison populations, including author-scene and mechanism supplements; no parameter/menu expansion',
        reuse_counts=dict(collections.Counter(r['method'] for r in rows if r['reuse'])),gap_counts=dict(collections.Counter(r['method'] for r in rows if not r['reuse'])),
        chen='current valid H/Forest/narrow runs already preserve h=execution=.15; reuse in place, never revive invalid old geometry port',
        C1='retain all frozen historical C1 trajectories and scenario-dependent controller periods; no new C1 policy or window retuning',
        acceleration='existing vector lidar reused with prior certificate; no new geometric memo applicable to these external controllers',new_training=0))
    dump(D/'COMPARISON_LINKS.json',rows);dump(D/'ACTIVE_NATIVE_CASES.json',cases)
    pf=register(cases,'native_v1');print(json.dumps(dict(rows=len(rows),new=len(cases),reuse=sum(r['reuse'] for r in rows),preflight=pf['summary']),ensure_ascii=False))

if __name__=='__main__':main()
