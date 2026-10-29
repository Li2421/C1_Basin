"""Single-merger coordinator, certification precedes all formal screening."""
from dispatch import *
from validation import certify,no_passage
from acceleration import accelerated_case
import traceback

def main():
 from unit_checks import main as units
 units()
 own=os.environ.get('SLURM_JOB_ID')
 while any(j['name'].startswith('brd1_')and j['job_id']!=own for j in jobs()):time.sleep(10)
 if list((HERE/'raw').glob('*.json')):merge('recovery',[dict(rollout_uid=p.stem)for p in(HERE/'raw').glob('*.json')])
 freeze=read(HERE/'DESIGN_FREEZE.json')
 assert all(sha(HERE/name)==expected for name,expected in freeze['numerical_files'].items()),'HARD_STOP frozen numerical definition changed'
 accelerated=certify();make=accelerated_case if accelerated else case
 no_passage(make);dump(HERE/'CERTIFICATION_READY.json',dict(complete=True,accelerated=accelerated))
 while not(HERE/'MAIN_READY').exists():
  status('等价认证完成，等待正式执行器',0,1,'尚未开始正式Oracle筛选；定义已冻结。');time.sleep(30)
 from screening import main as screen
 screen(make)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  dump(HERE/'PIPELINE_ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc()))
  with(HERE/'GAPS.md').open('a')as f:f.write('\n流程异常，保留所有原始结果与pending frontier：'+repr(exc)+'\n')
  raise
