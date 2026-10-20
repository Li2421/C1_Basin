from common import *
import subprocess,re
from zoneinfo import ZoneInfo
def main():
 cases=read(HERE/'gap_timing_CASES.json');submission=read(HERE/'gap_timing_SUBMISSION.json');job=submission['job'];assigned={i:[]for i in range(18)}
 for claim in (HERE/'gap_timing/claims').glob('*.json'):
  q=read(claim);assigned[q['worker']].append(read(HERE/'gap_timing/raw'/f"{q['case']['rollout_uid']}.json"))
 startup=[]
 for i in range(18):
  r=subprocess.run(['scontrol','show','job',job+'_'+str(i)],capture_output=True,text=True);match=re.search(r'StartTime=([0-9T:-]+)',r.stdout);start=datetime.fromisoformat(match.group(1)).replace(tzinfo=ZoneInfo('Asia/Singapore')) if match else datetime.fromisoformat(submission['utc']);end=datetime.fromisoformat(read(HERE/'gap_timing'/f'worker_{i}.json')['utc']);over=max(0.,(end-start).total_seconds()-sum(x['elapsed_seconds']for x in assigned[i]));startup.append(dict(worker=i,wall_bound_seconds=over,source='Slurm StartTime to completion minus measured episode wall time'if match else 'submission-to-completion conservative wall upper bound'))
 write(HERE/'MPC_COLD_OVERHEAD_AUDIT.json',dict(utc=utc(),workers=startup,definition='Conservative per-worker cold CPU upper bound from one-CPU wall startup, plus episode setup wall bound, plus first20 physical-step CPU.'))
 startup_bound=max(x['wall_bound_seconds']for x in startup)
 proof=read(HERE/'GAP_EXACT_PREFIX_PROOF.json')['passed'];out={}
 for n,H in [(2,2000),(10,8000),(20,16000)]:
  estimates={};valid=True
  for name in ['DM1','DM3']:
   rows=[read(HERE/'gap_timing/raw'/f"{c['rollout_uid']}.json") for c in cases if c['N']==n and c['base_controller_name']==name];means=[np.mean(r['warmed_step_CPU_seconds']) if r['warmed_step_CPU_seconds'] else None for r in rows];errors=[r['measurement_error'] for r in rows];ok=all(x is not None and np.isfinite(x) for x in means) and not any(errors);valid=valid and ok;step=max(means) if ok else None;cold=max(r['cold_CPU_seconds']+max(0.,r['elapsed_seconds']-r['cold_CPU_seconds']-sum(r['warmed_step_CPU_seconds'])) for r in rows)+startup_bound;hours=(step*64*(5+12/7 if proof else 9)*H+18*cold)/3600 if ok else None;estimates[name]=dict(two_state_warmed_CPU_step_seconds=means,worst_step_seconds=step,cold_CPU_seconds=cold,predicted_main_CPU_hours=hours,errors=errors)
  total=sum(v['predicted_main_CPU_hours'] for v in estimates.values()) if valid else None;out[str(n)]=dict(H=H,instances=estimates,predicted_total_CPU_hours=total,budget_CPU_hours=432,exact_prefix_proved=proof,eligible=bool(valid and total<=432),reason='within preregistered computation budget' if valid and total<=432 else '计算预算缺口' if valid else '测时接口或数值缺口')
 write(HERE/'MPC_TIMING_REPORT.json',out);seal('MPC_TIMING_REPORT.receipt.json',[HERE/'MPC_TIMING_REPORT.json',HERE/'GAP_EXACT_PREFIX_PROOF.json',HERE/'MPC_COLD_OVERHEAD_AUDIT.json'],before_capability_labels=True)
if __name__=='__main__':main()
