"""Final read-only evidence checks plus single official journal merge."""
from native_io import *
import subprocess,collections

def main():
    cases=read(D/'ACTIVE_NATIVE_CASES.json');assert len(cases)==375
    assert all((D/'raw'/(c['rollout_uid']+'.json')).exists() for c in cases)
    assert read(D/'NEW_TRACE_AUDIT.json')['passed'] and read(D/'NEW_TRACE_AUDIT.json')['cases']==len(cases)
    assert read(D/'NATIVE_LIDAR_QUERY_CERTIFICATE.json')['passed']
    subprocess.run([sys.executable,'-B',str(D/'capture_environment.py'),'--check'],check=True)
    subprocess.run([sys.executable,'-B',str(D/'migrate_completed_journals.py')],check=True)
    subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,check=True)
    new=planner.preflight(D/'tasks/native_v1_planned.json');dump(D/'FINAL_CACHE_CHECK.json',new)
    reuse=planner.preflight(D/'tasks/reuse_planned.json');dump(D/'REUSE_CACHE_CHECK.json',reuse)
    assert new['summary']['exact_reusable']==375 and reuse['summary']['exact_reusable']==311
    links=read(D/'COMPARISON_LINKS.json');checked=[]
    # Preserve immutable history, including actual artifact bytes; never count
    # a stale UID with a missing or changed trace as reusable evidence.
    for i,ref in enumerate(links):
        p=Path(ref['old_raw']);assert sha(p)==ref['old_raw_sha256']
        r=read(p);assert sha(r['trace'])==r['trace_sha256']
        checked.append(ref['old_uid'])
        if i%30==0:dump(D/'FINAL_HISTORY_PROGRESS.json',dict(checked=i+1,total=len(links)))
    assert len(links)==686 and len(set(checked))==686
    result=dict(passed=True,new_rollouts=375,exact_reused_history=311,physical_method_records=686,
        historical_artifacts_verified=len(checked),source_hashes_verified=read(D/'NEW_TRACE_AUDIT.json')['source_hashes_verified'],
        new_steps_replayed=read(D/'NEW_TRACE_AUDIT.json')['steps'],new_calls_verified=read(D/'NEW_TRACE_AUDIT.json')['control_calls'],
        native_clock_lidar_queries_bytewise=read(D/'NATIVE_LIDAR_QUERY_CERTIFICATE.json')['queries'],
        new_policy_training=0,temporal_profile='native_callback_ZOH_original_score_clock_v1',shared_cpu_limit=18,
        original_c1_policy_unchanged=True,valid_chen_native_period_preserved=True,native_author_dynamics_claimed=False,
        initial_infrastructure_failures='4 interpreters stopped before controller creation because lock directory was missing; directory created, same requests resumed; original failure records retained',
        numerical_environment_binding_sha256=sha(D/'ENVIRONMENT_BINDING.json'))
    dump(D/'FINAL_CHECK.json',result)
    subprocess.run([sys.executable,'-B',str(D/'build_report.py')],check=True)
    subprocess.run([sys.executable,'-B',str(D/'publish_entry.py')],check=True)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
