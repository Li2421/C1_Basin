"""Versioned delivery adapters; frozen numerical implementations stay unchanged."""
from common import *

def main():
 src=HERE/'review.py';text=src.read_text();assert text.count("seal('NUMERICAL_COMPLETE.receipt.json'")==1
 text=text.replace("seal('NUMERICAL_COMPLETE.receipt.json'","seal('NUMERICAL_COMPLETE_B.receipt.json'")
 (HERE/'review_B.py').write_text(text)
 src2=HERE/'delivery_v2.py';text=src2.read_text()
 text=text.replace('无新训练、安装或git写','无新训练、无原实验环境的包安装/升级、无git写；本轮plot_env仅用于绘图')
 text=text.replace('B未决定时Sprog缺口，HZ4完整协议不可判定。','B已按用户选定的候选1执行；完整修正菜单包含Sprog。')
 (HERE/'delivery_B.py').write_text(text)
 seal('B_DELIVERY_ADAPTER_FROZEN.receipt.json',[HERE/'review.py',HERE/'review_B.py',HERE/'delivery_v2.py',HERE/'delivery_B.py'],numerical_formulas_unchanged=True,changes=['New numerical completion receipt name preserving A/C completion receipt','Presentation qualification for isolated plot environment and now-executed B'])
if __name__=='__main__':main()
