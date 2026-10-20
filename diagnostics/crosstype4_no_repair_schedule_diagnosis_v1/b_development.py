"""Outer held-type labeled-action results, never used to select checkpoints."""
from common import *
from critic_model import metrics
from methods import METHODS
from catalog_names import full_type
import csv

def main():
    with np.load(HERE/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:labels=z['labels'][...,0];types=z['types'];names=z['names'];indices=z['candidate_indices'];raw=z['labels']
    with np.load(HERE/'DEVELOPMENT_CHOICES.npz',allow_pickle=False) as z:choices={k:z[k] for k in ('self_schedule','no_progress','no_zero','legacy_fixed','no_controller')}
    with np.load(HERE/'DEVELOPMENT_PROBABILITIES.npz',allow_pickle=False) as z:p=z['full']
    ranking=metrics(p,raw,types,sorted(set(types.tolist())),indices);rows={};mapping={'pipeline':'self_schedule','no_progress':'no_progress','no_zero':'no_zero','legacy_fixed':'legacy_fixed','no_controller':'no_controller'}
    for t in sorted(set(types.tolist())):
        mask=types==t;r={}
        for key,source in mapping.items():
            picked=np.take_along_axis(labels,choices[source][...,None],-1)[...,0][mask];r[METHODS[key]]=dict(success_lower_pct=float(np.mean(picked==1)*100),success_upper_pct=float(np.mean(picked!=0)*100),unknown=int(np.sum(picked<0)),instances=int(mask.sum()))
        rows[t]=dict(methods=r,ranking=ranking[t])
    write(HERE/'DEVELOPMENT_RESULTS.json',dict(utc=utc(),evidence='DEVELOPMENT',outer_held_type_folds=19,rows=rows,scope='48 training states,seed0,only34 labeled actions; not full441 oracle',outer_results_used_for_selection=False))
    ranking_csv(rows);curve_paths=curves()
    seal('DEVELOPMENT_RESULTS.receipt.json',curve_paths+[HERE/n for n in ('RANKING_ACCURACY.csv','DEVELOPMENT_RESULTS.json','DEVELOPMENT_CHOICES.receipt.json','b_development.py','methods.py')])
def ranking_csv(rows):
    labels={'all':'全部可辨识动作之间','diagonal':'对角元之间','non_diagonal':'非对角对之间','cross':'对角元与非对角对之间','same_pair_different_time':'同对不同切换方式之间','zero_vs_nonzero':'含η=0的动作与其余动作之间'}
    with (HERE/'RANKING_ACCURACY.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['类型','比较','准确率','支持controller数','支持state数','可辨识对数']);writer.writeheader()
        for typ,r in rows.items():
            for key,v in r['ranking']['ranking'].items():writer.writerow({'类型':full_type(typ),'比较':labels[key],'准确率':v['accuracy'],'支持controller数':v['supported_controllers'],'支持state数':v['supported_states'],'可辨识对数':v['identifiable_pairs']})

def curves():
    from html import escape
    selected=read(HERE/'SELECTED_MODELS.json');paths=[];folder=HERE/'figures';folder.mkdir(exist_ok=True)
    with (HERE/'TRAINING_CURVES.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['held_type','branch','step','outer_nll','selected_step','source_only_selected']);writer.writeheader()
        for held,group in selected['outer'].items():
            histories={};chosen={}
            for mode in ('full','no_controller'):
                model=group['full'][mode] if mode=='full' else group['no_controller'];histories[mode]=read(Path(model['checkpoint_path']).parent/'HISTORY.json');chosen[mode]=model['step']
                for r in histories[mode]:writer.writerow(dict(held_type=full_type(held),branch=mode,step=r['step'],outer_nll=r['validation'][held]['nll'],selected_step=chosen[mode],source_only_selected=True))
            all_values=[r['validation'][held]['nll'] for h in histories.values() for r in h];lo=min(all_values);hi=max(all_values);span=max(hi-lo,.01);lo-=.1*span;hi+=.1*span;y=lambda v:460-(v-lo)/(hi-lo)*330;x=lambda step:90+(step-100)/1900*820
            body=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="570"><rect width="1000" height="570" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#172337">',f'<text x="70" y="40" font-size="22">{escape(full_type(held))}：留出类型NLL曲线</text>','<text x="70" y="72" font-size="14">开发证据；checkpoint由源类型内部折选择，外层曲线只作报告。</text>']
            for mode,color in [('full','#496b9c'),('no_controller','#287d68')]:
                pts=' '.join(f"{x(r['step'])},{y(r['validation'][held]['nll'])}" for r in histories[mode]);body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>');xx=x(chosen[mode]);body.append(f'<path d="M{xx} 120 V460" stroke="{color}" stroke-dasharray="5 5"/>')
            for tick in (100,500,1000,1500,2000):body.append(f'<text x="{x(tick)}" y="490" text-anchor="middle">{tick}</text>')
            for v in np.linspace(lo,hi,5):body.append(f'<text x="78" y="{y(v)+5}" text-anchor="end">{v:.3f}</text>')
            body.extend(['<text x="80" y="530" fill="#496b9c">完整分支</text>','<text x="300" y="530" fill="#287d68">不看controller分支；虚线为各分支源内选中步数</text>','</g></svg>']);p=folder/f'B_training_curve_{held}.svg';p.write_text(''.join(body));paths.append(p)
    paths.append(HERE/'TRAINING_CURVES.csv');return paths
if __name__=='__main__':main()

