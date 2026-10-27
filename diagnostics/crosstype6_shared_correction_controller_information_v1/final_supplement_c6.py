"""Supplement a requested chart without changing any sealed science/report."""
from common import *
import subprocess

def main():
    for p,h in read(HERE/'DELIVERY.receipt.json')['files'].items():assert sha(p)==h,p
    plot_python='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python'
    try:
        result=subprocess.run([plot_python,'-B',str(HERE/'plot_grouped_information_v2.py')],cwd=ROOT,capture_output=True,text=True,timeout=600)
        (HERE/'GROUPED_BARS_RUN.log').write_text(result.stdout+'\n'+result.stderr)
        gap=None if result.returncode==0 else dict(returncode=result.returncode,log='GROUPED_BARS_RUN.log')
    except subprocess.TimeoutExpired:
        gap=dict(reason='Optional plotting exceeded ten-minute walltime; numeric work unchanged')
    if gap:write(HERE/'GROUPED_BARS_OPTIONAL_GAP.json',dict(utc=utc(),**gap,numeric_work_unchanged=True))
    subprocess.run([sys.executable,'-B',str(HERE/'closing_audit_c6_final.py')],cwd=ROOT,check=True)
    assert read(HERE/'INTEGRITY_FINAL_COMPARISON.json')['passed']
    text='# 补充交付\n\nREPORT.md、HANDOFF.md、原始数值和全部冻结模型保持原封存版本。\n\n'
    if not gap:
        text+='按 prompt 补齐信息价值表的预设分组条形图：第一层相对只加安全投影、加入在线停滞触发、加入离线探测特征，以及类型信息经验参照，共四个面板；每项附原封存95% CI，空组标为N/A。\n\n[分组条形图](figures/information_value_grouped_bars_v2.png)；可编辑SVG见同名.svg。原有三组图和收据均保留。\n\n'
    else:text+='可选分组条形图存在缺口，见GROUPED_BARS_OPTIONAL_GAP.json；原有图和主线数值完整保留。\n\n'
    text+='补图之后再次执行结束完整性审计，保护核查通过，见INTEGRITY_FINAL_COMPARISON.json。未重训、未重跑rollout、未修改控制器或成功标准。\n'
    (HERE/'DELIVERY_SUPPLEMENT.md').write_text(text)
    paths=[HERE/n for n in ('DELIVERY_SUPPLEMENT.md','DELIVERY.receipt.json','NUMERIC_RESULTS.receipt.json','PLOTS.receipt.json','INTEGRITY_FINAL_COMPARISON.receipt.json','final_supplement_c6.py')]
    paths.append(HERE/('GROUPED_BARS_OPTIONAL_GAP.json' if gap else 'INFORMATION_GROUPED_BARS.receipt.json'))
    seal('DELIVERY_FINAL.receipt.json',paths,integrity_passed=True,original_sealed_science_unchanged=True)
    print('Final supplemental delivery sealed',flush=True)

if __name__=='__main__':main()
