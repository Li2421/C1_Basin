from common import *
from pipeline import call
import subprocess

def main():
 assert (HERE/'A_COMPLETE.receipt.json').exists()
 if (HERE/'PSTAR_DECISION.receipt.json').exists() and read(HERE/'PSTAR_DECISION.receipt.json')['authorized_rollouts']:assert (HERE/'B_COMPLETE.receipt.json').exists()
 with (HERE/'plot_env_run.log').open('w') as f:subprocess.run([str(HERE/'plot_env/bin/python'),'-B',str(HERE/'plot_curves.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 call('immutability.py');call('audit.py','END');call('delivery.py')
 p=HERE/'REPORT.md';text=p.read_text();old='无新训练、安装或git写';assert old in text;text=text.replace(old,'无新训练、无原实验环境的包安装/升级、无git写；本轮plot_env仅用于绘图');p.write_text(text+'\n\n'+(HERE/'PLOT_ENV_QUALIFICATION.md').read_text()+'\nN2历史选择/验收/相向测试的分cohort表另见N2_HISTORICAL_EVIDENCE_TABLE.md，避免把dev选择的分母与后续fresh acceptance混同。\n')
 p=HERE/'HANDOFF.md';p.write_text(p.read_text()+'\n独立plot_env仅用于渲染曲线，所有仿真/统计/DB继续用原Python。绘图缺依赖后的新包装入口analysis_worker_v2.py不更改原冻结数值实现；见PLOT_ENV_QUALIFICATION和FINALIZATION_ENVIRONMENT_FIX.receipt。\n')
 call('review.py');extra=HERE/('PSTAR_DECISION.receipt.json' if (HERE/'PSTAR_DECISION.receipt.json').exists() else 'PSTAR_UNDECIDED.json');seal('SUPPLEMENTARY_DELIVERY.receipt.json',[HERE/'NUMERICAL_COMPLETE.receipt.json',extra,HERE/'PREREGISTRATION.receipt.json',HERE/'FINALIZATION_ENVIRONMENT_FIX.receipt.json',HERE/'B_ANALYSIS_FROZEN.receipt.json',HERE/'N2_HISTORICAL_TABLE.receipt.json'],no_extra_rollouts=True)
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'FINALIZATION_ERROR.json',dict(utc=utc(),error=repr(ex)));raise
