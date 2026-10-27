"""Early numerical tables on complete scopes; never seal full main completion."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from complete_stage_records import collect_complete_stages
import importlib.util
import fcntl
import time
import csv


def main():
    with (HERE/'COMPLETED_SCENE_SNAPSHOT.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_receipt('COMPLETED_SCENE_ANALYSIS_CODE.receipt.json')
        os.nice(10)
        write(HERE/'COMPLETED_SCENE_SNAPSHOT_PID.json',dict(utc=utc(),pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),nice=os.nice(0)))
        while True:
            if (HERE/'MIGRATION_NUMBERS.receipt.json').exists():
                write(HERE/'COMPLETED_SCENE_SNAPSHOT_SKIPPED.json',dict(utc=utc(),reason='Final main numbers already available; do not duplicate work'));return
            if (HERE/'LOCAL_SAVED_DIAGNOSIS_ERROR.json').exists():
                write(HERE/'COMPLETED_SCENE_SNAPSHOT_SKIPPED.json',dict(utc=utc(),reason='Local saved diagnosis has an error; defer to main pipeline'));return
            done=HERE/'LOCAL_SAVED_DIAGNOSIS_EXIT.json'
            if done.exists():
                row=read(done)
                if row['reason']=='All prepared completed-source diagnoses available':break
                write(HERE/'COMPLETED_SCENE_SNAPSHOT_SKIPPED.json',dict(utc=utc(),reason=row['reason']));return
            time.sleep(30)
        spec=importlib.util.spec_from_file_location('sealed_v2_snapshot_statistics',HERE/'statistics.py')
        stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
        originals=read(HERE/'LOCAL_DIAGNOSIS_SCOPE.json')['originals']
        records,sources=collect_complete_stages(originals)
        census=stats.controller_census(records);arms=[(layer,m) for layer in stats.LAYERS for m in stats.FIXED]
        results={};flat=[]
        for scene in NAMES:
            fixed=stats.summarize(records,arms,[scene],census=census)
            entry=dict(fixed_full=fixed)
            if scene!='lit_circle_five':
                keys={(r['N'],r['state']) for r in records if r['scene']==scene and r['layer']=='distributed' and r['method']=='learned_scene'}
                subset=[r for r in records if r['scene']==scene and (r['N'],r['state']) in keys]
                learned_arms=[('distributed',m) for m in (*stats.FIXED,'learned_scene')]
                if scene in ('lit_narrow_passage','lit_doorway'):learned_arms += [('central',m) for m in (*stats.FIXED,'learned_scene')]
                entry['learned_evaluation']=stats.summarize(subset,learned_arms,[scene],census=census)
            else:entry['learned_evaluation']=dict(status='pending',reason='Circular-scene labels and learned evaluation still running')
            results[scene]=entry
            for key in ('fixed_full','learned_evaluation'):
                flat.extend(dict(scene=scene,scope=key,**r) for r in entry[key].get('statistics',[]))
            write(HERE/'COMPLETED_SCENE_SNAPSHOT_PROGRESS.json',dict(utc=utc(),last_scene=scene,new_rollouts=0))
        write(HERE/'COMPLETED_SCENE_RESULTS.json',dict(utc=utc(),evidence='DEVELOPMENT',results=results,
            scope='Complete four-scene baseline/fixed scopes and three complete learned-evaluation scopes only; not full first-stage completion',
            bootstrap_iterations=5000,weighting='Identical sealed main summarize/bootstrap; type/controller/state clusters and paired seeds',
            new_rollouts=0,new_training=0))
        write(HERE/'COMPLETED_SCENE_RESULT_SOURCES.json',sources)
        fields=('scene','scope','type','layer','method','success_rate','ci95','success_upper','upper_ci95','controllers')
        with (HERE/'COMPLETED_SCENE_RESULTS.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            writer.writerows({k:canonical(r[k]) if isinstance(r[k],(list,dict)) else r[k] for k in fields} for r in flat)
        seal('COMPLETED_SCENE_NUMBERS.receipt.json',[HERE/n for n in ['COMPLETED_SCENE_RESULTS.json','COMPLETED_SCENE_RESULTS.csv','COMPLETED_SCENE_RESULT_SOURCES.json','COMPLETED_SCENE_ANALYSIS_CODE.receipt.json']],new_rollouts=0,new_training=0,full_main_complete=False,bootstrap_iterations=5000)
        print('Complete-scope numerical snapshot sealed; full main pipeline still running',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        write(HERE/'COMPLETED_SCENE_SNAPSHOT_ERROR.json',dict(utc=utc(),error=repr(exc),traceback=traceback.format_exc()))
        raise
