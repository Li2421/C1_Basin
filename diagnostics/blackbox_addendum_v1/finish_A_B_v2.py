from common import *
import subprocess,time,traceback
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def call(script,*args):
    print('RUN',script,args,flush=True);subprocess.run([PY,'-B',str(HERE/script),*args],cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu'},check=True)
def main():
    job=read(HERE/'geometry_controls_SUBMISSION.json')['job']
    while subprocess.check_output(['squeue','-r','-h','-j',job],text=True).strip():time.sleep(10)
    assert all((HERE/'geometry_controls'/f'worker_{i}.json').exists() for i in range(18))
    if not (HERE/'geometry_controls_COMPLETE.receipt.json').exists():call('postflight.py','geometry_controls')
    call('analyze_A_v2.py')
    if read(HERE/'A_RESULTS.json')['conditional_main']['triggered']:write(HERE/'A_CONDITIONAL_PENDING.json',dict(utc=utc(),required=True));return
    call('progress_decision.py');call('analyze_B.py');write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase='D_prepare',status='READY',completed_parts=['F','C','A','B']))
if __name__=='__main__':
    try:main()
    except Exception:write(HERE/'RUN_STATUS.json',dict(utc=utc(),status='STOPPED',error=traceback.format_exc()));raise
