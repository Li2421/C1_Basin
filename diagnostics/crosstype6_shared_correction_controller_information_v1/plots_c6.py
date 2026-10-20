"""Optional final plots: dedicated plotting venv, sealed numeric inputs only."""
from core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import csv

def main():
    receipt=read(HERE/'NUMERIC_RESULTS.receipt.json')
    for p,h in receipt['files'].items():assert sha(p)==h,p
    assert '.venv-c1-plots' in sys.executable
    available={f.name for f in font_manager.fontManager.ttflist};font=next((f for f in ('Noto Sans CJK SC','Noto Serif CJK SC','Droid Sans Fallback') if f in available),'DejaVu Sans');plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'svg.fonttype':'none','font.size':8})
    results=read(HERE/'T1_T7_RESULTS.json');events=read(HERE/'EVENT_AND_CHOICES.json');methods=list(results['groups']['类型等权合并']['methods']);types=np.asarray(events['types']);steps=np.asarray(events['trigger_steps']);alltypes=sorted(set(types));folder=HERE/'figures';folder.mkdir(exist_ok=True);paths=[]
    with (HERE/'AXIS_TABLE.csv').open() as f:full_names=dict(zip(alltypes,[r['类型 / 分组'] for r in csv.DictReader(f)][:len(alltypes)]))
    def save(fig,name):
        for ext in ('svg','png'):
            path=folder/f'{name}.{ext}';fig.savefig(path,dpi=200,bbox_inches='tight');paths.append(path)
        plt.close(fig)
    fig,axes=plt.subplots(5,4,figsize=(16,17))
    for ax,t in zip(axes.flat,alltypes):
        for mi,color in [(1,'#167db7'),(2,'#df8a1e'),(4,'#36944a')]:
            values=steps[types==t,:,mi,:];values=values[values>=0]
            if len(values):ax.hist(values,bins=np.arange(0,751,50),histtype='step',linewidth=1.5,label=methods[mi],color=color)
        ax.set_title(full_names[t]);ax.set_xlabel('触发步');ax.set_ylabel('已触发局数');ax.set_xlim(0,700)
    axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.02),ncol=1)
    fig.suptitle('各类型在线停滞触发步分布（未触发局未进入直方图）');fig.tight_layout();save(fig,'trigger_step_by_type')
    fig,axes=plt.subplots(5,4,figsize=(16,17));theta_values=['0.9','0.93','0.95','0.968','0.98','0.0']
    for ax,t in zip(axes.flat,alltypes):
        for offset,mi,color in [(-.24,0,'#6c7280'),(0,1,'#167db7'),(.24,2,'#df8a1e')]:
            counts=results['selected_threshold_counts'][t][methods[mi]];values=np.asarray([counts.get(v,0) for v in theta_values]);ax.bar(np.arange(6)+offset,values/max(1,values.sum()),width=.23,label=methods[mi],color=color)
        ax.set_title(full_names[t]);ax.set_xticks(range(6),['.90','.93','.95','.968','.98','无效阈值'],rotation=30);ax.set_ylim(0,1);ax.set_ylabel('选中比例')
    axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.02),ncol=1);fig.suptitle('各类型选择的进度阈值（整局单η时阈值无作用）');fig.tight_layout();save(fig,'selected_threshold_by_type')
    groups=['类型等权合并','学习型','优化型','搜索型','只加安全投影 ≥ 98%','90–98%','只加安全投影 < 90%','探测中早停滞','探测中晚停滞','探测中不停滞','选中η对距人定较近半组','选中η对距人定较远半组'];fig,axes=plt.subplots(1,2,figsize=(15,9),sharey=True)
    for ax,key,title,color in zip(axes,('T2_online','T3_offline'),('加入在线进度信息','再加入700步离线探测特征'),('#167db7','#df8a1e')):
        for gi,g in enumerate(groups):
            r=results['groups'][g]['comparisons'].get(key)
            if r is None:continue
            point=r['point'];lo,hi=r['CI95'];ax.hlines(gi,lo,hi,color=color);ax.vlines([lo,hi],gi-.08,gi+.08,color=color);ax.plot(point,gi,'o',color=color)
        ax.axvline(0,color='#555',linewidth=.7);ax.set_title(title);ax.set_xlabel('相邻两层配对成功率差（百分点，95% CI）');ax.set_yticks(range(len(groups)),groups);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.tight_layout();save(fig,'information_value_by_group')
    write(HERE/'PLOTS_MANIFEST.json',dict(utc=utc(),python=sys.executable,font=font,files=[str(p) for p in paths],sealed_numbers_only=True));seal('PLOTS.receipt.json',[HERE/'PLOTS_MANIFEST.json',HERE/'plots_c6.py',HERE/'NUMERIC_RESULTS.receipt.json',*paths],optional=True,no_numerical_recomputation=True)

if __name__=='__main__':main()
