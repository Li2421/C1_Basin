"""Resume from completed artifacts; never repeat a finished request or model freeze."""
from common import *
import subprocess
import fcntl

PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'


def command(script,*args):
    p=subprocess.run([PYTHON,'-B',str(HERE/script),*args],cwd=ROOT)
    assert p.returncode==0,(script,args,p.returncode)


def merged(stage):
    p=HERE/f'{stage}_MERGE.json'
    return p.exists()and read(p)['returncode']==0


def main():
    with (HERE/'ADVANCE.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (HERE/'COMPLETE.json').exists():return
        # A currently running capability scheduler owns its batch and merger.
        if not merged('capability'):
            while True:
                proc=subprocess.run(['pgrep','-f','^'+PYTHON+' -B '+str(HERE/'schedule.py')+' capability$'],capture_output=True,text=True)
                # Relative-path manual scheduler is also recognized.
                other=subprocess.run(['pgrep','-f','^'+PYTHON+' -B diagnostics/cross_scene_correction_v1/schedule.py capability$'],capture_output=True,text=True)
                if not proc.stdout.strip()and not other.stdout.strip():break
                time.sleep(30)
            if not merged('capability'):command('schedule.py','capability')
        if not(HERE/'SELECTED_CONTROLLERS.json').exists():
            from stages import capability_select
            capability_select()
        for stage in ('pilot','main'):
            if not(HERE/f'{stage}_PREFLIGHT.receipt.json').exists():command('stages.py',stage)
            if not merged(stage):command('schedule.py',stage)
            if stage=='pilot'and not(HERE/'PILOT_TIMINGS.json').exists():
                from stages import pilot_cost
                pilot_cost()
        if not(HERE/'MODEL_FROZEN.receipt.json').exists():command('analyze.py')
        if not(HERE/'extrapolation_PREFLIGHT.receipt.json').exists():command('extrapolate.py','prepare')
        if not merged('extrapolation'):command('schedule.py','extrapolation')
        if not(HERE/'EXTRAPOLATION_RESULTS.json').exists():command('extrapolate.py','results')
        if not(HERE/'STATISTICS.json').exists():command('statistics.py')
        command('postflight.py');command('deliver.py')


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(HERE/'ADVANCE_ERROR.json',dict(utc=utc(),traceback=traceback.format_exc(),no_scientific_unknown_retried=True))
        raise
