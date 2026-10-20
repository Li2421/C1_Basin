"""B1–B5 development statistics, paired type/controller/state bootstrap."""
from common import *
from methods import METHODS
from catalog_names import full_type
import csv
COMPARISONS={'相对只加安全投影':('pipeline','safety'),'相对去掉不修正':('pipeline','no_zero'),'相对去掉按进度':('pipeline','no_progress'),'相对不看controller分支':('pipeline','no_controller')}

def bootstrap(labels,types,group,rng,reps=5000):
    lower=(labels==1).mean(-1);upper=(labels!=0).mean(-1);ts=sorted(set(types[group].tolist()));result=[];upper_result=[]
    if not ts:return None,None
    for rep in range(reps+1):
        values=[];uppers=[]
        for typ in (ts if rep==0 else rng.choice(ts,len(ts),replace=True)):
            choices=np.flatnonzero(group&(types==typ));controllers=choices if rep==0 else rng.choice(choices,len(choices),replace=True);cv=[];uv=[]
            for i in controllers:
                states=np.arange(labels.shape[1]) if rep==0 else rng.integers(labels.shape[1],size=labels.shape[1]);cv.append(lower[i,states].mean(0));uv.append(upper[i,states].mean(0))
            values.append(np.mean(cv,0));uppers.append(np.mean(uv,0))
        result.append(np.mean(values,0));upper_result.append(np.mean(uppers,0))
    return np.asarray(result),np.asarray(upper_result)

def summary(values):return dict(point=float(values[0]*100),CI95=np.quantile(values[1:]*100,[.025,.975]).tolist())

def average_ranks(x):
    x=np.asarray(x);order=np.argsort(x,kind='stable');r=np.empty(len(x),float);k=0
    while k<len(x):
        end=k+1
        while end<len(x) and x[order[end]]==x[order[k]]:end+=1
        r[order[k:end]]=(k+end-1)/2+1;k=end
    return r

def main():
    assert (HERE/'NEW_EVAL_BUNDLE.receipt.json').exists()
    with np.load(HERE/'NEW_EVAL_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    labels=data['labels'];uids=data['uids'];names=data['names'];types=data['types'];methods=data['methods'].tolist();ix={k:methods.index(k) for k in methods};safety=(labels[:,:,ix['safety']]==1).mean((1,2));pipeline=(labels[:,:,ix['pipeline']]==1).mean((1,2));high=safety>=.98;low=safety<.90
    cap=read(D/'crosstype3_unseen_controller_schedule_v1/UNSEEN_CAPABILITY.json');cap_lookup={r['controller']:r['safety_success_rate'] for r in cap['rows']}
    with (D/'crosstype3_unseen_controller_schedule_v1/ZOO_MANIFEST.csv').open() as source:
        for row in csv.DictReader(source):
            try:v=float(row['safety_capability_rate'])
            except (ValueError,TypeError):continue
            cap_lookup.setdefault(row['controller'],v/100 if v>1 else v)
    cap_high=np.asarray([cap_lookup.get(str(n),-1)>=.98 for n in names]);groups={t:types==t for t in sorted(set(types.tolist()))};groups.update({'类型等权合并':np.ones(len(names),bool),'只加安全投影 ≥ 98% 的 controller':high,'只加安全投影 < 90% 的 controller':low,'跨类型3能力门只加安全投影 ≥ 98% 的 controller':cap_high});rng=np.random.default_rng(202610194502);results={}
    choices=read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states'];zero=np.zeros((74,24),bool);anyzero=np.zeros_like(zero);prog=np.zeros_like(zero)
    for r in choices:
        a=r['methods']['pipeline'];i=list(names).index(r['controller']);j=r['state_index'];ea=bool(np.all(np.asarray(a['eta_a'])==0));eb=bool(np.all(np.asarray(a['eta_b'])==0));zero[i,j]=ea and eb;anyzero[i,j]=ea or eb;prog[i,j]=a['switch_mode']=='progress'
    for label,mask in groups.items():
        print('B bootstrap',label,int(mask.sum()),flush=True);samples,uppers=bootstrap(labels,types,mask,rng)
        if samples is None:results[label]=dict(controllers=0,methods={},comparisons={},gap='empty development subgroup');continue
        row=dict(controllers=int(mask.sum()),types=len(set(types[mask].tolist())),methods={},comparisons={})
        def type_mean(values):return float(np.mean([values[mask&(types==t)].mean() for t in sorted(set(types[mask].tolist()))]))
        for method in methods:row['methods'][METHODS[method]]=dict(**summary(samples[:,ix[method]]),unknown_upper=summary(uppers[:,ix[method]]))
        for label2,(a,b) in COMPARISONS.items():
            v=samples[:,ix[a]]-samples[:,ix[b]];row['comparisons'][label2]=summary(v)
            # Pointwise sharp paired uncertainty bounds, holding known labels.
            same=uids[:,:,ix[a]]==uids[:,:,ix[b]];lo=np.where(same,0,(labels[:,:,ix[a]]==1).astype(float)-(labels[:,:,ix[b]]!=0).astype(float)).mean((1,2));hi=np.where(same,0,(labels[:,:,ix[a]]!=0).astype(float)-(labels[:,:,ix[b]]==1).astype(float)).mean((1,2));row['comparisons'][label2]['unknown_bounds']=[100*type_mean(lo),100*type_mean(hi)]
        human=np.maximum(samples[:,ix['human500']],samples[:,ix['human_progress']]);row['comparisons']['相对较好人定时间表']=summary(samples[:,ix['pipeline']]-human);row['comparisons']['相对较好人定时间表']['best_human_reselected_each_bootstrap']=True
        row['harm_count']=int(np.sum(mask&((safety-pipeline)>=.1-1e-12)));row['choice_fractions']=dict(zero_zero=type_mean(zero.mean(1)),other_containing_zero=type_mean((anyzero&~zero).mean(1)),any_zero=type_mean(anyzero.mean(1)),progress=type_mean(prog.mean(1)))
        row['events']={METHODS[m]:{k:type_mean(data[k][:,:,ix[m]].mean((1,2)))*100 for k in ('collision','timeout','unknown')} for m in methods};results[label]=row
    def goal_B1(key):
        r=results[key]
        return dict(available=r['controllers']>0,achieved=r['controllers']>0 and r['comparisons']['相对只加安全投影']['point']>=-1 and r['harm_count']==0,controllers=r['controllers'],gain=r.get('comparisons',{}).get('相对只加安全投影'),harm_count=r.get('harm_count'),zero_zero_fraction=r.get('choice_fractions',{}).get('zero_zero'))
    b2=results['只加安全投影 < 90% 的 controller'];rx=average_ranks(safety);ry=average_ranks(zero.mean(1));corr=float(np.corrcoef(rx,ry)[0,1]) if rx.std()>0 and ry.std()>0 else None
    goals=dict(B1_new_states=goal_B1('只加安全投影 ≥ 98% 的 controller'),B1_crosstype3_capability=goal_B1('跨类型3能力门只加安全投影 ≥ 98% 的 controller'),B2=dict(available=b2['controllers']>0,achieved=b2['controllers']>0 and b2['comparisons']['相对去掉不修正']['CI95'][0]>-1,controllers=b2['controllers'],gain=b2.get('comparisons',{}).get('相对去掉不修正'),mistaken_zero_zero_fraction=b2.get('choice_fractions',{}).get('zero_zero')),B3=results['类型等权合并']['comparisons']['相对较好人定时间表'],B4={k:results['类型等权合并']['comparisons'][k] for k in ('相对去掉按进度','相对去掉不修正')},B5=dict(gain=results['类型等权合并']['comparisons']['相对不看controller分支'],spearman_average_ranks=corr))
    result=dict(utc=utc(),evidence='DEVELOPMENT',bootstrap=5000,seed=202610194502,groups=results,goals=goals,capability_missing=[str(n) for n in names if n not in cap_lookup],conditional_groups_fixed_to_observed_safety=True,no_unseen_confirmatory_claim=True);write(HERE/'B_RESULTS.json',result)
    with (HERE/'B_CONTROLLER_RESULTS.csv').open('w') as f:
        fields=['controller','type','safety_pct','pipeline_pct','zero_zero_fraction','other_containing_zero_fraction','progress_fraction','harm_ge10pp'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i,n in enumerate(names):writer.writerow(dict(controller=str(n),type=full_type(types[i]),safety_pct=100*safety[i],pipeline_pct=100*pipeline[i],zero_zero_fraction=zero[i].mean(),other_containing_zero_fraction=(anyzero[i]&~zero[i]).mean(),progress_fraction=prog[i].mean(),harm_ge10pp=int(safety[i]-pipeline[i]>=.1-1e-12)))
    make_tables(result);make_scatter(names,types,safety,zero.mean(1),corr)
    seal('B_RESULTS.receipt.json',[HERE/n for n in ('B_RESULTS.json','B_CONTROLLER_RESULTS.csv','AXIS_TABLE.md','AXIS_TABLE.csv','B_EVENT_TABLE.csv','figures/B_zero_choice_vs_safety.svg','b_statistics.py','NEW_EVAL_BUNDLE.receipt.json')],all_development=True)

def make_tables(result):
    base=read(HERE/'BASE_PROTOCOL.json');all_types=sorted(set(base['types'].values()));methods=list(METHODS.values());lines=['# 19类型开发检验','', '本轮主方法 = 学习双 η 含不修正·可选按进度。所有类型均已被研究者使用；留一类型是本轮模型的训练划分，不构成未见类型确认性检验。','', '成功率 % [95% CI]。其他8个学习型训练类型按prompt不做新状态评估，标记N/A，不把历史不同状态结果混入本表。','', '| 类型 / 分组 | '+' | '.join(methods)+' |','|---|'+'---|'*len(methods)]
    for label in [*all_types,'类型等权合并','只加安全投影 ≥ 98% 的 controller','只加安全投影 < 90% 的 controller','跨类型3能力门只加安全投影 ≥ 98% 的 controller']:
        row=result['groups'].get(label);cells=[]
        for method in methods:
            if row is None or method not in row['methods']:cells.append('N/A')
            else:r=row['methods'][method];cells.append(f"{r['point']:.2f} [{r['CI95'][0]:.2f}, {r['CI95'][1]:.2f}]")
        lines.append('| '+full_type(label)+' | '+' | '.join(cells)+' |')
    lines.extend(['','碰撞率、超时率、unknown、伤害数与不修正/进度选择比例见 B_EVENT_TABLE.csv 与 B_RESULTS.json。','', '不加安全投影不在本轮第四节的新状态动作清单中，不额外生成该对照。','', '跨类型3能力门的旧训练实例缺乏可比数值率者保留缺口，见 B_RESULTS.json capability_missing。'])
    (HERE/'AXIS_TABLE.md').write_text('\n'.join(lines)+'\n')
    with (HERE/'AXIS_TABLE.csv').open('w') as f:
        fields=['类型 / 分组',*methods,'证据等级','完整菜单平均遗憾','完整菜单最大遗憾','相对全程η10灾难数'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for label in [*all_types,'类型等权合并','只加安全投影 ≥ 98% 的 controller','只加安全投影 < 90% 的 controller','跨类型3能力门只加安全投影 ≥ 98% 的 controller']:
            row=result['groups'].get(label);record={'类型 / 分组':full_type(label),'证据等级':'DEVELOPMENT','完整菜单平均遗憾':'N/A：未执行441动作完整菜单','完整菜单最大遗憾':'N/A：未执行441动作完整菜单','相对全程η10灾难数':'N/A：本轮无全程η10新状态标签'}
            for method in methods:
                if row is None or method not in row['methods']:record[method]='N/A：按prompt不评估或分组为空'
                else:
                    r=row['methods'][method];record[method]=f"{r['point']:.2f} [{r['CI95'][0]:.2f}, {r['CI95'][1]:.2f}]"
            writer.writerow(record)

    with (HERE/'B_EVENT_TABLE.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['group','method','collision_pct','timeout_pct','unknown_pct','harm_count','zero_zero_fraction','other_containing_zero_fraction','progress_fraction']);writer.writeheader()
        for group,r in result['groups'].items():
            for m,v in r.get('events',{}).items():writer.writerow(dict(group=group,method=m,collision_pct=v['collision'],timeout_pct=v['timeout'],unknown_pct=v['unknown'],harm_count=r['harm_count'],zero_zero_fraction=r['choice_fractions']['zero_zero'],other_containing_zero_fraction=r['choice_fractions']['other_containing_zero'],progress_fraction=r['choice_fractions']['progress']))

def make_scatter(names,types,safety,zero,corr):
    from html import escape
    body=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="720"><rect width="1100" height="720" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#172337">','<text x="65" y="40" font-size="24">学习双 η 含不修正·可选按进度：不修正选择与安全投影成功率</text>',f'<text x="65" y="70" font-size="16">开发证据；74个controller；平均秩 Spearman = {corr if corr is not None else "N/A"}</text>']
    colors=['#126e82','#aa3453','#5d54a4','#bc7626','#31805a','#96728e','#657b2c','#b25a3a','#44709f','#71818a','#3d474f'];palette=dict(zip(sorted(set(types.tolist())),colors))
    for v in (0,.2,.4,.6,.8,1):
        x=100+v*680;y=580-v*450;body.extend([f'<path d="M100 {y} H780 M{x} 130 V580" stroke="#e0e5ec" fill="none"/>',f'<text x="85" y="{y+5}" text-anchor="end">{v*100:.0f}</text>',f'<text x="{x}" y="605" text-anchor="middle">{v*100:.0f}</text>'])
    for n,t,x,y in zip(names,types,safety,zero):body.append(f'<circle cx="{100+x*680}" cy="{580-y*450}" r="5" fill="{palette[t]}" opacity=".8"><title>{escape(str(n))}: 只加安全投影 {100*x:.2f}%，选择不修正 {100*y:.2f}%</title></circle>')
    for k,(t,color) in enumerate(palette.items()):body.extend([f'<circle cx="820" cy="{140+k*29}" r="5" fill="{color}"/>',f'<text x="835" y="{145+k*29}" font-size="12">{escape(full_type(t))}</text>'])
    body.extend(['<text x="420" y="650" text-anchor="middle" font-size="17">只加安全投影成功率（%）</text>','<text x="25" y="350" transform="rotate(-90 25 350)" text-anchor="middle" font-size="17">选择 (0, 0) 的比例（%）</text>','</g></svg>']);p=HERE/'figures/B_zero_choice_vs_safety.svg';p.parent.mkdir(exist_ok=True);p.write_text(''.join(body))
if __name__=='__main__':main()
