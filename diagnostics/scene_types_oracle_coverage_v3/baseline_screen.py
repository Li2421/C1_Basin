"""User-added fixed paper baselines, strictly after the four-column phase."""
from dispatch import *
from papers import make,version,prepare,NAMES
from baseline_analysis import analyze
import traceback

OUT=HERE/'paper_after_four_column'

def save_labels(cases,profile,phase):
    rows=[dict(rollout_uid=c['rollout_uid'],profile=profile['id'],method=profile['method'],
        scene=c['scene'],type_id=c['type_id'],level=next(e['level']for e in read(HERE/'SCENES.json')['entries']if e['scene']['name']==c['scene']),
        state_index=c['state_index'],solo=c['solo'],phase=phase)for c in cases]
    path=OUT/'labels'/f"{profile['id']}_{version(profile)}_{phase}.json"
    dump(path,rows)

def main():
    assert(HERE/'family_4_complete.json').exists()
    protocol=read(OUT/'PROFILES_FROZEN.json');scenes=read(HERE/'SCENES.json')['entries']
    statuses=[]
    for ordinal,profile in enumerate(protocol['profiles']):
        profile=prepare(profile)
        method=profile['method'];tag=version(profile);prefix=f'papers_{ordinal:02d}_{tag}'
        try:
            # Base ladder/state0 pilots are an exact subset of the full joint
            # cohort, not a different benchmark or an extra tuning set.
            pilot=[make(e['scene'],0,profile,'paper_fixed_profile_pilot')for e in scenes if e['level']==0]
            save_labels(pilot,profile,'pilot')
            rows=execute(pilot,prefix+'_pilot',worker_entry='worker_papers.py')
            setup=[r for r in rows if r.get('setup_error')]
            dump(OUT/f"pilot_{profile['id']}_{tag}.json",dict(requests=len(pilot),complete=len(rows)==len(pilot),
                 setup_errors=[dict(uid=r['rollout_uid'],error=r['setup_error'])for r in setup],
                 numeric_or_action_errors=sum(bool(r.get('paper_error_events'))for r in rows)))
            # Missing optional runtime interfaces are implementation gaps, not
            # hard-scene evidence. Other methods and C1 families still proceed.
            if rows and all(r.get('setup_error')and r['setup_error']['type']in ('ImportError','ModuleNotFoundError','AttributeError','NameError')for r in rows):
                statuses.append(dict(profile=profile['id'],complete=False,gap='All pilots fail to construct the port; inspect saved setup errors.'))
                analyze();continue
            solo=[]
            for e in scenes:
                for agent in range(e['scene']['config']['num_agents']):
                    solo.append(make(e['scene'],0,profile,'paper_single_robot_diagnostic',solo=agent))
            save_labels(solo,profile,'solo')
            solo_rows=execute(solo,prefix+'_solo',worker_entry='worker_papers.py')
            analyze()
            joint=[make(e['scene'],i,profile,'paper_fixed_profile_comparison')for e in scenes for i in range(8)]
            save_labels(joint,profile,'joint')
            joint_rows=execute(joint,prefix+'_joint',worker_entry='worker_papers.py')
            statuses.append(dict(profile=profile['id'],version=tag,complete=len(joint_rows)==len(joint)and len(solo_rows)==len(solo),
                                 joint_complete=len(joint_rows),joint_expected=len(joint),solo_complete=len(solo_rows),solo_expected=len(solo)))
            dump(OUT/'EXECUTION_STATUS.json',statuses);analyze()
        except Exception as exc:
            if 'HARD_STOP'in str(exc)or(HERE/'PROTECTION_VIOLATION.jsonl').exists():raise
            item=dict(profile=profile['id'],complete=False,error=repr(exc),traceback=traceback.format_exc())
            statuses.append(item);dump(OUT/'EXECUTION_STATUS.json',statuses)
            with(HERE/'GAPS.md').open('a')as f:f.write('\n论文固定对照缺口：'+NAMES[method]+' '+repr(exc)+'。其他方法继续。\n')
            analyze()
    result=dict(complete=len(statuses)==len(protocol['profiles'])and all(x['complete']for x in statuses),
                phase_finished=True,profiles=statuses,parameter_search=False,after_family4=True)
    dump(OUT/'COMPLETE.json',result);analyze();return result

if __name__=='__main__':main()
