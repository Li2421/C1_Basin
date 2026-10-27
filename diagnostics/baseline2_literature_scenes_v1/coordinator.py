"""Resume exact missing requests; keep two CPU shares for local coordination.

No numerical unknown is retried. Only absent raw records after an ended job
are reclaimed. Every submission still uses submit.py and its exact preflight.
"""
from experiment import *
import subprocess
import time
import traceback
import fcntl

PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
SCENES=list(NAMES)

def command(file,*args):
    if (file=='submit.py' and len(args)>=2 and args[1]=='flow_certified.sbatch'
            and args[0].endswith('lit_circle_five')):
        if (HERE/'TILED_PARENT_DEPLOYMENT.receipt.json').exists():args=(args[0],'flow_tiled.sbatch',*args[2:])
        elif (HERE/'CONTROLLER_INTERIOR_DEPLOYMENT.receipt.json').exists():args=(args[0],'flow_interior.sbatch',*args[2:])
    print(utc(),file,*args,flush=True)
    if file in ('design_flow.py','design_background.py','freeze_scene_choices.py') or file=='learning_design.py' and args[0]=='plan':
        command('db_merge.py')
    with (HERE/'logs'/'coordinator_commands.log').open('a') as f:
        p=subprocess.run([PY,'-B',str(HERE/file),*map(str,args)],stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError((file,args,p.returncode))

def queue():
    lines=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-r','-o','%i|%C|%b']).decode().splitlines()
    return {line.split('|')[0].split('_')[0] for line in lines},sum(int(line.split('|')[1]) for line in lines)

def active(stage):
    jobs,_=queue()
    return any(str(read(p)['job_id']).split(';')[0] in jobs for p in [*HERE.glob(f'{stage}_SUBMISSION.json'),*HERE.glob(f'{stage}_RESUBMISSION_*.json')])

def exists(stage):return (HERE/f'{stage}_PREFLIGHT.receipt.json').exists()
def done(stage):
    if not exists(stage):return False
    cases=read(HERE/f'{stage}_MISSING_CASES.json')
    raw={p.stem for p in (HERE/stage/'raw').glob('*.json')}
    return all(c['rollout_uid'] in raw for c in cases)

def launch(stage,maximum=8,script='flow_certified.sbatch'):
    if done(stage) or active(stage):return
    _,used=queue();spare=16-used
    if spare<1:return
    if script=='flow_certified.sbatch' and not (HERE/stage/'STREAM_QUEUE.receipt.json').exists():command('stream_queue.py','build',stage)
    command('stream_queue.py','recover',stage)
    command('submit.py',stage,script,min(maximum,spare))

def fill_spare_primary_capacity():
    """Additional workers claim only untouched groups of an active queue.

    Fresh eligible stages are submitted first by the ordinary dependency loop.
    This step changes allocation only, and never changes requests or seeds.
    """
    _,used=queue();spare=16-used
    if spare<1:return
    candidates=[]
    for scene in SCENES:
        cap=('capability_v2_' if scene=='lit_narrow_passage' else 'capability_')+scene
        candidates.extend((s,'flow_certified.sbatch') for s in
                          (cap,'safety_'+scene,'fixed_'+scene,
                           'learning_labels_'+scene,'learning_eval_'+scene))
    for scene in SCENES:
        candidates.extend((s,'flow_extended.sbatch') for s in
                          ('pilot_learning_labels_'+scene,'learning_eval_'+scene+'_with_bc'))
    for stage,script in candidates:
        index=HERE/stage/'GROUP_INDEX.json'
        if not index.exists() or not active(stage) or done(stage):continue
        # Completed groups retain their claim directories. Existing owners
        # also retain theirs, so none of their physical cases are reassigned.
        available=sum(not (HERE/stage/'claims'/e['group']).exists()
                      for e in read(index))
        if available:
            command('submit.py',stage,script,min(spare,available))
            return

def _coordinate():
    failures=0;laststatus=0
    while not (HERE/'COORDINATOR_STOP').exists():
        try:
            for scene in SCENES:
                cap=('capability_v2_' if scene=='lit_narrow_passage' else 'capability_')+scene
                if not done(cap):
                    if exists(cap):launch(cap)
                    continue
                if not (HERE/f'{cap}_QUALIFICATION.json').exists():command('phase_reports.py','capability',cap)
                safety='safety_'+scene
                if not exists(safety):command('design_flow.py',safety,scene)
                if not done(safety):launch(safety);continue
                if not (HERE/f'BASELINE_USABILITY_{scene}.json').exists():command('analyze_baseline.py',scene)
                fixed='fixed_'+scene
                if not exists(fixed):command('design_flow.py',fixed,scene)
                if not done(fixed):launch(fixed,10);continue
                gap=HERE/f'LEARNING_GAP_{scene}.json';labels='learning_labels_'+scene
                if not gap.exists() and not exists(labels):command('learning_design.py','plan',scene)
                if gap.exists():continue
                if not done(labels):launch(labels,6);continue
                if not (HERE/f'LEARNING_BUNDLE_{scene}.receipt.json').exists():command('learning_design.py','assemble',scene)
                if not (HERE/f'FROZEN_SCENE_CRITIC_{scene}.receipt.json').exists():
                    _,used=queue()
                    if used<16 and not active('critic_'+scene):command('submit.py','critic_'+scene,'train_critic.sbatch',1,1)
                    continue
                evaluation='learning_eval_'+scene
                if not exists(evaluation):command('freeze_scene_choices.py',scene)
                if not done(evaluation):launch(evaluation,6)
            # Background runs never displace pending primary-stage submissions.
            # They also complete section C's white-box reference before BC.
            _,used=queue()
            if used<16:
                for scene in SCENES:
                    if not done('fixed_'+scene):continue
                    if not exists('nominal_'+scene):command('design_background.py',scene)
                    launch('whitebox_'+scene,2)
                    launch('nominal_'+scene,4,'flow_nominal.sbatch')
            from pilot_advance import advance
            advance()
            # The optional paper equations are independent of the main line.
            # They use only capacity left after all currently eligible work.
            if exists('paper_secondary'):launch('paper_secondary',2,'paper_secondary.sbatch')
            fill_spare_primary_capacity()
            if time.monotonic()-laststatus>=900:
                command('phase_reports.py','status');laststatus=time.monotonic()
            write(HERE/'COORDINATOR_STATE.json',dict(utc=utc(),all_blackbox_L1=all(done('fixed_'+s) for s in SCENES),all_background=all(done('nominal_'+s) and done('whitebox_'+s) for s in SCENES),errors=failures))
        except Exception:
            failures+=1;message=traceback.format_exc();print(message,flush=True)
            write(HERE/f'COORDINATOR_ERROR_{time.time_ns()}.json',dict(utc=utc(),traceback=message))
            # A programming/identity error requires inspection; do not loop retries.
            break
        time.sleep(30)

def main():
    with (HERE/'COORDINATOR_INSTANCE.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another coordinator owns this experiment; no second scheduler started.',flush=True);return
        _coordinate()

if __name__=='__main__':main()
