"""Sequential phase transitions; always preflight and seal before submission."""
from core import *
import subprocess

def resources():
    lines=subprocess.check_output(['squeue','-r','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%C'],text=True).splitlines();running=[]
    for line in lines:
        job,name,status,cpus=line.split('|')
        if status in ['RUNNING','COMPLETING','CONFIGURING']:running.append(dict(job=job,name=name,status=status,cpus=int(cpus)))
    return running

def submit(phase,workers=20):
    active=resources();available=20-sum(r['cpus'] for r in active);assert available>0,'No authorized CPU available; wait, do not oversubscribe'
    path=HERE/('probe.sbatch' if phase=='probe' else 'rollout.sbatch');cmd=['sbatch',f'--array=0-{workers-1}%{min(workers,available)}',str(path)]+([str(workers)] if phase=='probe' else [phase,str(workers)]);out=subprocess.check_output(cmd,text=True);write(HERE/(phase+'_SUBMISSION.json'),dict(utc=utc(),cmd=cmd,result=out,active_before_submission=active,cpu_limit=20,gpu_shards_limit=2,max_concurrency=min(workers,available),preflight_sha256=sha(HERE/f'{phase}_PREFLIGHT.receipt.json')));print(out,flush=True)

def pilot():
    from design import initial
    from phase_design import create
    assert not (HERE/'pilot_SUBMISSION.json').exists();initial();create('pilot');(HERE/'main_protocol.json').write_bytes((HERE/'pilot_protocol.json').read_bytes());seal('PILOT_SUBMISSION_READY.receipt.json',[HERE/'pilot_PREFLIGHT.receipt.json',HERE/'run.py',HERE/'student.py',HERE/'coordinates.py',HERE/'episode_extended.py',HERE/'episode_frozen.py',HERE/'pipeline.py',HERE/'actions.py',HERE/'core.py'],before_submission=True);submit('pilot')

def probe():
    from postflight import merge
    merge('pilot')
    if not read(HERE/'CAPABILITY_GATE.json')['eligible']:
        write(HERE/'NO_ELIGIBLE_CONTROLLERS.json',dict(utc=utc(),status='INCONCLUSIVE',reason='No eligible student'));return
    from probes import design
    design();submit('probe')

def mainphase():
    from probes import freeze
    freeze()
    from shared_rollout_db.src.cache_writer import append_journal
    from shared_rollout_db.src.rollout_db import connect
    journal=append_journal([read(p) for p in sorted((HERE/'probe/raw').glob('*.json'))],'exp_teacher_student_macflow_v1b_probe','unlabeled_provenance');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals')
    cmd=[sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment','exp_teacher_student_macflow_v1b_probe'];r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);(HERE/'probe_MERGE.log').write_text(r.stdout+r.stderr);assert r.returncode==0
    with connect(True) as db:
        n=db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',('exp_teacher_student_macflow_v1b_probe',)).fetchone()[0];assert n==0
        sources=[dict(r) for r in db.execute('SELECT * FROM source_file WHERE experiment_uid=?',('exp_teacher_student_macflow_v1b_probe',))]
    write(HERE/'probe_JOURNAL_RECEIPT.json',dict(utc=utc(),unlabeled_rows=len(list((HERE/'probe/raw').glob('*.json'))),task_rollouts=0,sources=sources,classification='Unlabeled provenance, SUMMARY_ONLY; no fabricated task labels',official_command=cmd,journal=str(journal)))
    from freeze_choices import main
    main()
    from phase_design import create
    create('main');seal('MAIN_IMPLEMENTATION_FROZEN.receipt.json',[HERE/'main_PREFLIGHT.receipt.json',HERE/'CHOICES_FROZEN.receipt.json',HERE/'TRAINING_FROZEN.receipt.json',HERE/'run.py',HERE/'pipeline.py',HERE/'episode_extended.py',HERE/'episode_frozen.py',HERE/'student.py',HERE/'actions.py',HERE/'core.py'],before_submission=True);submit('main')

if __name__=='__main__':globals()[sys.argv[1]]()
