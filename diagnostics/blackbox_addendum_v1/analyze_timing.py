from common import *
def main():
 cases=read(HERE/'gap_timing_CASES.json');proof=read(HERE/'GAP_EXACT_PREFIX_PROOF.json')['passed'];out={}
 for n,H in [(2,2000),(10,8000),(20,16000)]:
  estimates={};valid=True
  for name in ['DM1','DM3']:
   rows=[read(HERE/'gap_timing/raw'/f"{c['rollout_uid']}.json") for c in cases if c['N']==n and c['base_controller_name']==name];means=[np.mean(r['warmed_step_CPU_seconds']) if r['warmed_step_CPU_seconds'] else None for r in rows];errors=[r['measurement_error'] for r in rows];ok=all(x is not None and np.isfinite(x) for x in means) and not any(errors);valid=valid and ok;step=max(means) if ok else None;cold=max(r['cold_CPU_seconds'] for r in rows);hours=(step*64*(5+12/7 if proof else 9)*H+18*cold)/3600 if ok else None;estimates[name]=dict(two_state_warmed_CPU_step_seconds=means,worst_step_seconds=step,cold_CPU_seconds=cold,predicted_main_CPU_hours=hours,errors=errors)
  total=sum(v['predicted_main_CPU_hours'] for v in estimates.values()) if valid else None;out[str(n)]=dict(H=H,instances=estimates,predicted_total_CPU_hours=total,budget_CPU_hours=432,exact_prefix_proved=proof,eligible=bool(valid and total<=432),reason='within preregistered computation budget' if valid and total<=432 else '计算预算缺口' if valid else '测时接口或数值缺口')
 write(HERE/'MPC_TIMING_REPORT.json',out);seal('MPC_TIMING_REPORT.receipt.json',[HERE/'MPC_TIMING_REPORT.json',HERE/'GAP_EXACT_PREFIX_PROOF.json'],before_capability_labels=True)
if __name__=='__main__':main()
