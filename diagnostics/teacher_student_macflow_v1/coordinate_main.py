from core import *
from shared_rollout_db.src import cache_writer
from shared_rollout_db.src.rollout_db import connect
import subprocess,time

def main():
    assert Path(cache_writer.__file__).resolve()==ROOT/'shared_rollout_db/src/cache_writer.py'
    expected=len(read(HERE/'probe_CASES.json'))
    while len(list((HERE/'probe/raw').glob('*.json')))<expected or not all((HERE/f'imports_probe_{i}.json').exists() for i in range(17)):
        live=subprocess.check_output(['squeue','-h','-j','21386','-o','%T'],text=True).strip()
        if not live:raise RuntimeError('Probe batch ended before all records; inspect, preserve unknown')
        time.sleep(20)
    from probes import freeze
    freeze()
    # Store unlabeled provenance through the official journal/merger, never fake task outcomes.
    journal=cache_writer.append_journal([read(p) for p in sorted((HERE/'probe/raw').glob('*.json'))],'exp_teacher_student_macflow_v1_probe','unlabeled_provenance')
    result=subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment','exp_teacher_student_macflow_v1_probe'],cwd=ROOT,capture_output=True,text=True);(HERE/'probe_MERGE.log').write_text(result.stdout+result.stderr);assert result.returncode==0
    with connect(True) as db:
        n=db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',('exp_teacher_student_macflow_v1_probe',)).fetchone()[0];assert n==0
        sources=[dict(r) for r in db.execute('SELECT * FROM source_file WHERE experiment_uid=?',('exp_teacher_student_macflow_v1_probe',))]
    write(HERE/'probe_JOURNAL_RECEIPT.json',dict(utc=utc(),unlabeled_rows=expected,task_rollouts=0,sources=sources,classification_note='Official merger treats unlabeled provenance as SUMMARY_ONLY; rows_ambiguous means no task outcome, not a missing probe or an imputed outcome. Raw hashes/R5 receipt are the unlabeled cache.'))
    from freeze_choices import main as choices
    choices()
    from phase_design import create
    create('main')
    while not (HERE/'INFERENCE_TIMING.json').exists():time.sleep(10)
    # Freeze worker import provenance and current implementation before either submission.
    import shared_rollout_db.src.cache_writer as official
    assert Path(official.__file__).resolve()==ROOT/'shared_rollout_db/src/cache_writer.py'
    seal('MAIN_IMPLEMENTATION_FROZEN.receipt.json',[HERE/'run.py',HERE/'episode_frozen.py',HERE/'episode_extended.py',HERE/'pipeline.py',HERE/'actions.py',HERE/'student.py',HERE/'core.py',HERE/'main_PREFLIGHT.receipt.json',HERE/'teacher_PREFLIGHT.receipt.json',HERE/'CHOICES_FROZEN.receipt.json'],before_submission=True)
    a=subprocess.check_output(['sbatch','--array=0-9%10',str(HERE/'rollout.sbatch'),'main','10'],text=True);b=subprocess.check_output(['sbatch','--array=0-7%8',str(HERE/'rollout.sbatch'),'teacher','8'],text=True)
    write(HERE/'MAIN_SUBMISSION.json',dict(utc=utc(),student=a,teacher=b,CPU_shards=18,GPU_shards=0));print(a+b,flush=True)
if __name__=='__main__':main()
