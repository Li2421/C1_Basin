"""Audit omitted physical/task constants without altering any execution.

The shorter-budget example only censors existing first-success times. It is
not a new simulation dataset and never enters the benchmark or model fitting.
"""
from collections import Counter
import numpy as np
from .experiment import HERE,read,write,connect,sha
from .factorial import execution_constants
from .evaluate import csvwrite


def run():
    p=read(HERE/'protocol.json');summary=[];examples=[]
    with connect(True) as db:
        for scene in p['environments']:
            cfg=p['environments'][scene]['config']['config']
            state=next(s for s in p['states'] if s['scenario']==scene and s['study_split']=='train')
            stored=np.load(HERE/'features'/state['state_uid']/'analytic0.npz')['entity_globals']
            assert np.all(stored==0)
            counts=Counter()
            for case in read(HERE/'batches/source_q1/cases.json'):
                if case['scene']!=scene or case['split']!='train' or not case['controller_name'].startswith('analytic'):continue
                r=db.execute('SELECT success,episode_length,numerical_failure FROM rollout WHERE rollout_uid=?',(case['rollout_uid'],)).fetchone()
                if not r['success'] or r['numerical_failure']:continue
                counts['original_successful_attempts']+=1
                for horizon in (100,300,500):
                    if horizon>=cfg['max_steps']:continue
                    success_before=int(r['episode_length']<=horizon)
                    counts[f'success_after_budget_{horizon}']+=success_before
                    if not success_before:
                        examples.append(dict(scene=scene,state_uid=case['state_uid'],controller=case['controller_name'],
                            eta_index=case['eta_index'],seed=case['seed'],rollout_uid=case['rollout_uid'],
                            original_max_steps=cfg['max_steps'],shorter_scoring_budget=horizon,
                            original_first_success_step=r['episode_length'],original_success=1,success_before_short_budget=0))
            summary.append(dict(scene=scene,actual_constants={k:cfg.get(k) for k in ('dt','max_speed','max_steps',
                'terminate_on_deadlock','progress_window_seconds','deadlock_hold_seconds')},
                encoded_globals=stored.tolist(),restored_globals=execution_constants(scene).tolist(),**counts))
    write(HERE/'environment_information_audit.json',dict(summaries=summary,
        source_only=True,new_rollouts=0,labels_changed_in_database=False,
        missing_information='All15 existing global channels are zero, including known physical speed/time and task termination constants',
        prefix_example='Success-by-budget differs for identical physical/controller/eta input when the outcome time budget changes. Counts are first-success censoring of cached original successful trajectories, not recertified short-horizon rollouts.',
        limitation='This proves the representation is not sufficient over arbitrary task budgets. It does NOT establish ambiguity on the four fixed benchmark distributions, whose geometries can implicitly identify their constants, or prove that omitted constants caused the observed LOSO failure.',
        feature_definition='Original15global channel order; physical/task constants restored, Flow commitment channel remains0, reset monitor history empty, no scene-ID.',
        code_sha256=sha(__file__)))
    csvwrite(HERE/'environment_budget_alias_examples.csv',examples)
    print('Environment information audit complete; no rollout changes',summary,flush=True)


if __name__=='__main__':run()
