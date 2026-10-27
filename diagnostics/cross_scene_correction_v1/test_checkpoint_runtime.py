"""Compare canonical execution with two independent-process restores, bit for bit."""
from common import *
from checkpoint_store import CheckpointStore,CheckpointPause,TARGET
import numpy as np
import subprocess

PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'


def child(folder,phase):
    c=read(folder/'case.json')
    if phase=='reference':
        from runtime_new import run
        result,arrays=run(c)
    else:
        from runtime_checkpointed import run
        forced={'pause37':(37,),'pause83':(83,),'resumed':()}[phase]
        checkpoint=CheckpointStore(c,folder=folder/'snapshots',forced_steps=forced)
        try:result,arrays=run(c,checkpoint)
        except CheckpointPause as exc:
            write(folder/f'{phase}.json',dict(paused=True,receipt=str(exc),
                resumed_from_next_step=None if checkpoint.resumed_from is None else checkpoint.resumed_from['next_step']));return
        assert not forced,('EXPECTED_PAUSE_DID_NOT_HAPPEN',phase)
    write(folder/f'{phase}.json',result)
    np.savez_compressed(folder/f'{phase}.npz',**arrays)


def main():
    folder=HERE/'checkpoint_validation'/str(time.time_ns());folder.mkdir(parents=True)
    pilot=read(HERE/'pilot_CASES.json');rows=[]
    for name in ('xsc_gap_n2','xsc_gap_n4','xsc_gap_n6'):
        c=copy.deepcopy(next(c for c in pilot if c['state']['scene']==name and c['state']['index']==0
            and c['requested_action']=='eta13'and c['controller_payload']['nominal']['controller']=='prioritized_astar0'))
        # Explicit interface test, not an experiment label or changed case identity.
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
        assert canonical(reference)==canonical(resumed),('RESULT_DIFF',name)
        with np.load(sample/'reference.npz',allow_pickle=False)as a,np.load(sample/'resumed.npz',allow_pickle=False)as b:
            assert set(a.files)==set(b.files)
            fields=[]
            for key in a.files:
                if key=='step_seconds':continue
                assert np.array_equal(a[key],b[key]),('TRACE_DIFF',name,key)
                fields.append(key)
        rows.append(dict(scene=name,horizon=160,pauses=[37,83],independent_processes=4,
            verified_restart_next_steps=[0,37,83],
            exact_result=True,exact_trace_fields=fields,artifacts=str(sample),test_only_no_rollout_db_writes=True))
        print('Checkpoint replay passed:',name,flush=True)
    paths=[HERE/n for n in ('runtime_new.py','runtime_checkpointed.py','checkpoint_store.py','build_checkpoint_runtime.py','test_checkpoint_runtime.py')]
    write(HERE/'CHECKPOINT_REPLAY_CHECK.json',dict(utc=utc(),passed=True,rows=rows,
        implementation_sources={str(p):sha(p)for p in paths},scope_exact_rollout_uid=TARGET,
        canonical_runtime_unchanged=sha(HERE/'runtime_new.py')==read(HERE/'CHECKPOINT_BUILD.json')['baseline_sha256'],
        source_inverse_transform_exact=read(HERE/'CHECKPOINT_BUILD.json')['inverse_text_transform_exact'],
        timing_fields_excluded=['elapsed_seconds','step_seconds'],
        no_controller_physics_rng_or_success_criterion_changes=True,
        unit_test_horizon_not_a_scientific_label=True,official_journal_writes=0))


if __name__=='__main__':
    if len(sys.argv)>1:child(Path(sys.argv[2]),sys.argv[3])
    else:main()
