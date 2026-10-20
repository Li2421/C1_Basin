from common import *
from pipeline import call

def main():
 assert (HERE/'A_COMPLETE.receipt.json').exists();assert (HERE/'CONTROLLER_RADIUS_CURVES.png').exists() and (HERE/'CONTROLLER_RADIUS_CURVES.pdf').exists()
 if (HERE/'PSTAR_DECISION.receipt.json').exists() and read(HERE/'PSTAR_DECISION.receipt.json')['authorized_rollouts']:assert (HERE/'B_COMPLETE.receipt.json').exists()
 call('immutability.py');call('audit.py','END');call('delivery_v2.py')
 p=HERE/'REPORT.md';text=p.read_text();old='无新训练、安装或git写';assert old in text;text=text.replace(old,'无新训练、无原实验环境的包安装/升级、无git写；本轮plot_env仅用于绘图');p.write_text(text+'\n\n'+(HERE/'GEOMETRY_SCOPE.md').read_text()+'\n\n'+(HERE/'PLOT_ENV_QUALIFICATION.md').read_text()+'\nN2历史选择/验收/相向测试的分cohort表另见N2_HISTORICAL_EVIDENCE_TABLE.md，避免混同dev选择与fresh acceptance。\n')
 p=HERE/'HANDOFF.md';p.write_text(p.read_text()+'\n独立plot_env仅渲染JSON曲线，所有仿真/统计/DB用原Python。最终入口analysis_worker_v3.py使用presentation dictionary修正版delivery_v2.py，未改统计或物理。首两次最终作业错误保留，见DELIVERY_PRESENTATION_FIX.receipt。群体中位数几何脆弱有v8个体例外，见GEOMETRY_SCOPE.md。\n')
 call('review.py');extra=HERE/('PSTAR_DECISION.receipt.json' if (HERE/'PSTAR_DECISION.receipt.json').exists() else 'PSTAR_UNDECIDED.json');seal('SUPPLEMENTARY_DELIVERY.receipt.json',[HERE/'NUMERICAL_COMPLETE.receipt.json',extra,HERE/'PREREGISTRATION.receipt.json',HERE/'FINALIZATION_ENVIRONMENT_FIX.receipt.json',HERE/'DELIVERY_PRESENTATION_FIX.receipt.json',HERE/'B_ANALYSIS_FROZEN.receipt.json',HERE/'N2_HISTORICAL_TABLE.receipt.json',HERE/'GEOMETRY_SCOPE.md'],no_extra_rollouts=True)
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'FINALIZATION_ERROR.json',dict(utc=utc(),error=repr(ex)));raise
