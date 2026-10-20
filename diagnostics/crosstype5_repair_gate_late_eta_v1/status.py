"""Metadata-only rollout progress and observed throughput, separate from evidence conclusions."""
from common import *
from resource_guard import queue
import time

def update(stage=None):
    now=time.time();phases={}
    for name in ('training','coverage','evaluation'):
        path=HERE/f'{name}_BUDGET.json'
        if not path.exists():continue
        budget=read(path);paths=list((HERE/name/'raw').glob('*.json'));recent=sum(p.stat().st_mtime>now-300 for p in paths);cap=budget['actual_execution_cap'];remaining=max(0,cap-len(paths));complete=(HERE/f'{name}_COMPLETE.receipt.json').exists()
        phases[name]=dict(real_completed=len(paths),real_execution_cap=cap,remaining=remaining,completed=complete,real_last_5_minutes=recent,remaining_hours_at_recent_throughput=remaining/(recent*12) if recent and not complete else None)
    result=dict(utc=utc(),stage=stage,queue=queue(),phases=phases,generator_fits_completed=len(list((HERE/'schedule_generator_fits').glob('*/COMPLETE.receipt.json'))),critic_fits_completed=len(list((HERE/'critic_fits').glob('*/*/COMPLETE.receipt.json'))),final_delivery_completed=(HERE/'DELIVERY.receipt.json').exists(),ETA_is_operational_not_scientific_result=True)
    write(HERE/'STATUS.json',result);lines=['# 跨类型5运行进度','',f"UTC {result['utc']}；当前阶段 {stage}。运行上限为全用户24 CPU/5 GPU，实际排队/占用见STATUS.json。"]
    for name,r in phases.items():lines+=['',f"{name}: 真实执行 {r['real_completed']} / {r['real_execution_cap']}；近5分钟 {r['real_last_5_minutes']} 条；此阶段是否完成 {r['completed']}。"]
    lines+=['',f"时间表生成器拟合 {result['generator_fits_completed']} / 191；三seed critic拟合 {result['critic_fits_completed']} / 382。",'', '完成时间随Slurm调度与后续精确去重数量更新。科学结论以最终REPORT.md/results.json为准，补标进度不代表成功率提升。']
    (HERE/'STATUS.md').write_text('\n'.join(lines)+'\n');return result
if __name__=='__main__':print(json.dumps(clean(update(sys.argv[1] if len(sys.argv)>1 else None)),indent=2))
