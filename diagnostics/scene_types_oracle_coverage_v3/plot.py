"""Scientific coverage curves from frozen JSON only; no controller imports."""
import guard
from guard import HERE
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def main():
    font=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family']=font.get_name();plt.rcParams['svg.fonttype']='none'
    data=json.loads((HERE/'COVERAGE.json').read_text())
    folder=HERE/'figures';folder.mkdir(exist_ok=True)
    files=[]
    for family in data['families']:
        fi=family['index'];name=family['name']
        stopped=[g for g in family['cases_detail']if g.get('screened_out')]
        end=min(g['searched']for g in stopped)if stopped else 64
        if stopped:name+='（提前筛选停止；后续覆盖未知）'
        fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
        ax.plot([x['K']for x in family['curve'][:end]],[x['covered']for x in family['curve'][:end]],linewidth=2)
        ax.set(xlabel='实际搜索候选数上限',ylabel=f"已找到成功的案例数（总计{family['cases']}）",xlim=(1,64),ylim=(0,family['cases']),title=name+(''if data['complete']else'（已验证覆盖下界，搜索未完成）'))
        ax.grid(alpha=.25)
        p=folder/f'family_{fi}_all.svg';fig.savefig(p);files.append(str(p));plt.close(fig)
        fig,axes=plt.subplots(5,4,figsize=(15,15),constrained_layout=True)
        for ax,t in zip(axes.flat,data['by_type']):
            ax.plot(range(1,end+1),t['curves'][str(fi)][:end],linewidth=1.5)
            denominator=t.get('denominator_by_family',{}).get(str(fi),t['denominator'])
            ax.set(title=t['name'],xlim=(1,64),ylim=(0,denominator))
            ax.set_xlabel('搜索候选数上限');ax.set_ylabel(f"覆盖数 /{denominator}");ax.grid(alpha=.2)
        for ax in list(axes.flat)[17:]:ax.axis('off')
        fig.suptitle(name+(''if data['complete']else'（当前已验证覆盖，搜索尚未完成）'),fontsize=13)
        p=folder/f'family_{fi}_by_type.svg';fig.savefig(p);files.append(str(p));plt.close(fig)
    (HERE/'FIGURES.json').write_text(json.dumps(dict(files=files,source='COVERAGE.json',zero_rollouts=True),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
