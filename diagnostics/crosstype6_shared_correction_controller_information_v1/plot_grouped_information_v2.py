"""Supplemental grouped bars from already sealed numeric evidence only."""
from core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import textwrap

def main():
    assert '.venv-c1-plots' in sys.executable
    for p,h in read(HERE/'NUMERIC_RESULTS.receipt.json')['files'].items():assert sha(p)==h,p
    r=read(HERE/'T1_T7_RESULTS.json');oracle=read(HERE/'TYPE_INFORMATION_ORACLE_RESULTS.json')
    methods=list(r['groups']['类型等权合并']['methods'])
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ('Noto Sans CJK SC','Noto Serif CJK SC') if f in available),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'svg.fonttype':'none','font.size':8})
    groups=['类型等权合并','学习型','优化型','搜索型','只加安全投影 ≥ 98%','90–98%','只加安全投影 < 90%','探测中早停滞','探测中晚停滞','探测中不停滞','选中η对距人定较近半组','选中η对距人定较远半组']
    comparisons=['T1_vs_Safety','T2_online','T3_offline',None]
    titles=[methods[0]+' 相对 '+methods[7],methods[1]+' 相对 '+methods[0],methods[2]+' 相对 '+methods[1],'类型信息经验参照：同类其他实例按 state 选，相对其他类型按 state 选']
    fig,axes=plt.subplots(1,4,figsize=(25,11),sharey=True)
    for ax,key,title,color in zip(axes,comparisons,titles,('#67717d','#167db7','#df8a1e','#38845a')):
        for gi,g in enumerate(groups):
            v=r['groups'][g]['comparisons'].get(key) if key else oracle['groups'][g]
            if not v or 'point' not in v:
                ax.text(0,gi,'N/A',va='center',fontsize=8);continue
            point=v['point'];lo,hi=v['CI95']
            ax.barh(gi,point,height=.55,color=color,alpha=.35)
            ax.hlines(gi,lo,hi,color=color,linewidth=1.4)
            ax.vlines([lo,hi],gi-.08,gi+.08,color=color,linewidth=1.4)
            ax.plot(point,gi,'o',color=color,markersize=4)
        ax.axvline(0,color='#555',linewidth=.7)
        ax.set_title('\n'.join(textwrap.wrap(title,25)),fontsize=9)
        ax.set_xlabel('配对成功率差（百分点，95% CI）')
        ax.set_yticks(range(len(groups)),groups);ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle('controller 信息价值：预设分组条形图（开发证据；5000 次配对 bootstrap）',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));paths=[]
    for ext in ('png','svg'):
        path=HERE/'figures'/f'information_value_grouped_bars_v2.{ext}'
        fig.savefig(path,dpi=200,bbox_inches='tight');paths.append(path)
    plt.close(fig)
    write(HERE/'INFORMATION_GROUPED_BARS_MANIFEST.json',dict(utc=utc(),python=sys.executable,font=font,files=[str(p) for p in paths],sealed_numbers_only=True,original_plots_preserved=True,empty_groups_marked_NA=True))
    seal('INFORMATION_GROUPED_BARS.receipt.json',[HERE/'INFORMATION_GROUPED_BARS_MANIFEST.json',HERE/'plot_grouped_information_v2.py',HERE/'NUMERIC_RESULTS.receipt.json',*paths],no_numerical_recomputation=True)

if __name__=='__main__':main()
