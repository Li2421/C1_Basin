"""CBS continuation regression plus initial-only serialization of new geometries."""
from common import *
from scene_environment import NewEnv,NewNominal
from legacy_states import LiteratureConfig
from scenes import make_new
import numpy as np
import pickle
import subprocess

PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'


def main():
    original=read(HERE/'CHECKPOINT_REPLAY_CHECK.json');assert original['passed']
    for name,h in original['implementation_sources'].items():assert sha(name)==h
    folder=HERE/'checkpoint_validation'/f'planners_{time.time_ns()}';folder.mkdir(parents=True)
    pilot=read(HERE/'pilot_CASES.json');rows=[]
    for name in ('xsc_gap_n2','xsc_gap_n4','xsc_gap_n6'):
        c=copy.deepcopy(next(c for c in pilot if c['state']['scene']==name and c['state']['index']==0
            and c['requested_action']=='eta13'and c['controller_payload']['nominal']['controller']=='conflict_based_search0'))
        c['scenario_payload']['config']['max_steps']=160
        sample=folder/name;sample.mkdir();write(sample/'case.json',c)
        for phase in ('reference','pause37','pause83','resumed'):
            with(sample/f'{phase}.log').open('w')as log:
                p=subprocess.run([PYTHON,'-B',str(HERE/'test_checkpoint_runtime.py'),'child',str(sample),phase],cwd=ROOT,stdout=log,stderr=log)
            assert p.returncode==0,(name,phase,p.returncode)
        reference=read(sample/'reference.json');resumed=read(sample/'resumed.json')
        assert read(sample/'pause37.json')['resumed_from_next_step']is None
        assert read(sample/'pause83.json')['resumed_from_next_step']==37
        assert resumed['operational_resume']['resumed_from']['next_step']==83
        for key in ('elapsed_seconds','operational_resume'):
            reference.pop(key,None);resumed.pop(key,None)
        assert canonical(reference)==canonical(resumed),('CBS_RESULT_DIFF',name)
        fields=[]
        with np.load(sample/'reference.npz',allow_pickle=False)as a,np.load(sample/'resumed.npz',allow_pickle=False)as b:
            assert set(a.files)==set(b.files)
            for key in a.files:
                if key=='step_seconds':continue
                assert np.array_equal(a[key],b[key]),('CBS_TRACE_DIFF',name,key)
                fields.append(key)
        rows.append(dict(scene=name,controller='conflict_based_search0',horizon=160,verified_restart_next_steps=[0,37,83],
            exact_result=True,exact_trace_fields=fields,artifacts=str(sample)))
        print('CBS checkpoint replay passed:',name,flush=True)
    initial=[];selected=read(HERE/'SELECTED_CONTROLLERS.json')
    for scene in read(HERE/'SCENES.json'):
        if scene['family']=='literature':continue
        for kind in('prioritized_astar','conflict_based_search'):
            ctl=selected[scene['name']].get(kind)
            if ctl is None:continue
            c=make_new(scene,ctl,scene['states'][0],actions()[0])
            env=NewEnv(c['state'],LiteratureConfig(**c['scenario_payload']['config']));actor=NewNominal(ctl)
            restored,nominal=pickle.loads(pickle.dumps((env,actor),protocol=pickle.HIGHEST_PROTOCOL))
            for key in('positions','velocities','goals','walls'):assert np.array_equal(getattr(env,key),getattr(restored,key))
            assert canonical(actor.__dict__)==canonical(nominal.__dict__)
            initial.append(dict(scene=scene['name'],controller=ctl['controller'],physical_steps=0,
                initial_geometry_and_actor_exact=True,extrapolation_initial_definition_only=scene['family']=='extrapolation'))
    sources={**original['implementation_sources'],**{str(HERE/n):sha(HERE/n)for n in('checkpoint_scope.py','test_checkpoint_planners.py')}}
    write(HERE/'CHECKPOINT_PLANNER_SCOPE_CHECK.json',dict(utc=utc(),passed=True,implementation_sources=sources,
        base_prioritized_astar_proof_sha256=sha(HERE/'CHECKPOINT_REPLAY_CHECK.json'),cbs_replay_rows=rows,
        initial_serialization_checks=initial,scope='New-scene prioritized_astar/conflict_based_search only; no legacy runtime change.',
        extrapolation_multiagent_actions_or_results_observed=False,official_journal_writes=0,
        no_controller_physics_rng_or_success_criterion_changes=True))


if __name__=='__main__':main()
