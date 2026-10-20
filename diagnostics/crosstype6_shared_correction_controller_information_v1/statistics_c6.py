"""Numeric T1--T7 only. No plotting imports or Markdown prerequisite."""
from common import *
from menu_c6 import METHODS
from catalog_names import full_type
import csv
PAIRS=((0,7),(0,6),(1,0),(2,1),(0,3),(1,4),(1,5))
PAIR_NAMES=('T1_vs_Safety','T1_vs_human','T2_online','T3_offline','T4_threshold','T5_state_trigger','T7_vs_C5')
LEARNING={'Flow','gaussian','deterministic','mixture','diffusion','chunking','recurrent','cvae','ring_ippo','ring_ippo_conservative','transformer_bc'}
SEARCH={'prioritized_astar','conflict_based_search'}

def summary(values):return dict(point=float(values[0]*100),CI95=np.quantile(values[1:]*100,[.025,.975]))

def macro(values,types,mask):
    tt=sorted(set(types[mask]));return float(np.mean([values[mask&(types==t)].mean() for t in tt])) if tt else None

def bootstrap(arrays,types,mask,rng,B=5000):
    tt=sorted(set(types[mask]));K=arrays.shape[-1]
    if not tt:return None
    out=np.zeros((B+1,K),np.float64);out[0]=np.mean([arrays[mask&(types==t)].mean((0,1)) for t in tt],0);draw=rng.choice(tt,(B,len(tt)))
    for t in tt:
        pool=np.flatnonzero(mask&(types==t));reps,_=np.where(draw==t)
        for start in range(0,len(reps),100):
            rr=reps[start:start+100];cc=rng.choice(pool,(len(rr),len(pool)));ss=rng.integers(arrays.shape[1],size=(len(rr),len(pool),arrays.shape[1]));v=arrays[cc[:,:,None],ss].mean((1,2));np.add.at(out,rr+1,v/len(tt))
    return out

def grouping(d):
    types=d['types'];names=d['names'];safety=(d['labels'][:,:,7]==1).mean((1,2));groups={t:types==t for t in sorted(set(types))};groups.update({'类型等权合并':np.ones(len(names),bool),'学习型':np.isin(types,list(LEARNING)),'优化型':~np.isin(types,list(LEARNING|SEARCH)),'搜索型':np.isin(types,list(SEARCH)),'只加安全投影 ≥ 98%':safety>=.98,'90–98%':(safety>=.90)&(safety<.98),'只加安全投影 < 90%':safety<.90});raw=read(HERE/'PROBE_FEATURE_RAW.json');stall={};gaps=[]
    for name in names:
        rr=[r for r in raw if r['controller']==name and r['arm']=='safety' and not r['failed']];ss=[r['first_stall_step'] for r in rr if r['first_stall_step'] is not None]
        stall[str(name)]='gap' if not rr else ('early' if np.median(ss)<200 else 'late') if ss else 'none'
    for key,value in [('探测中早停滞','early'),('探测中晚停滞','late'),('探测中不停滞','none')]:groups[key]=np.asarray([stall[str(n)]==value for n in names])
    base=read(HERE/'BASE_PROTOCOL.json');distance={n:[] for n in names};radius=np.asarray([.375,.5,.375]);links={(r['controller'],r['state_index'],r['seed']):r['methods'][METHODS[1]] for r in read(HERE/'evaluation_ACTION_LINKS.json')};unresolved=[]
    for r in read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states']:
        a=r['methods'][METHODS[1]];late=np.asarray(a['eta_b'],float)
        if a.get('late_source','given')=='requery':
            realized=[]
            for q in (0,1):
                p=HERE/'evaluation/raw'/f"{links[(r['controller'],r['state_index'],q)]}.json"
                rr=read(p) if p.exists() else None
                if rr is not None and rr.get('late_eta_requery') is not None:realized.append(rr['late_eta_requery']['eta_b'])
            if not realized:unresolved.append(dict(controller=r['controller'],state_index=r['state_index'],reason='late requery not realized; no invented eta'));continue
            value=float(np.mean([np.linalg.norm(np.r_[(np.asarray(a['eta_a'])-base['eta'][13])/radius,(np.asarray(b)-base['eta'][10])/radius]) for b in realized]))
        else:value=float(np.linalg.norm(np.r_[(np.asarray(a['eta_a'])-base['eta'][13])/radius,(late-base['eta'][10])/radius]))
        distance[r['controller']].append(value)
    means={str(n):float(np.mean(v)) for n,v in distance.items() if v};order=sorted(means,key=lambda n:(means[n],n));near=set(order[:len(order)//2]);far=set(order[len(order)//2:]);groups['选中η对距人定较近半组']=np.asarray([n in near for n in names]);groups['选中η对距人定较远半组']=np.asarray([n in far for n in names])
    write(HERE/'GROUP_MEMBERSHIP.json',dict(utc=utc(),groups={k:names[v].tolist() for k,v in groups.items()},stall_class=stall,eta_distance_controller_mean=means,eta_distance_unresolved=unresolved,eta_distance_wholly_unresolved_controllers=[str(n) for n in names if str(n) not in means],membership_fixed_descriptive=True))
    return groups

def main():
    assert (HERE/'NEW_EVAL_BUNDLE.receipt.json').exists()
    with np.load(HERE/'NEW_EVAL_BUNDLE.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    assert d['methods'].tolist()==list(METHODS);label=d['labels'];uids=d['uids'];types=d['types'];names=d['names'];lower=(label==1).mean(-1);upper=(label!=0).mean(-1);bound=[]
    for a,b in PAIRS:
        same=uids[:,:,a]==uids[:,:,b];bound.extend([np.where(same,0,(label[:,:,a]==1).astype(float)-(label[:,:,b]!=0)).mean(-1),np.where(same,0,(label[:,:,a]!=0).astype(float)-(label[:,:,b]==1)).mean(-1)])
    arrays=np.concatenate([lower,upper,np.stack(bound,-1)],-1);groups=grouping(d);rng=np.random.default_rng(2026102064);results={};samples={};theta={str(t):{} for t in sorted(set(types))}
    for r in read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states']:
        for method in METHODS[:5]:
            a=r['methods'][method];value=str(a.get('progress_threshold',0.));theta[r['type']].setdefault(method,{});theta[r['type']][method][value]=theta[r['type']][method].get(value,0)+1
    safety=lower[:,:,7].mean(1);online=lower[:,:,1].mean(1);first=lower[:,:,0].mean(1)
    for group,mask in groups.items():
        print('Numeric paired bootstrap',group,int(mask.sum()),flush=True);s=bootstrap(arrays,types,mask,rng)
        if s is None:results[group]=dict(controllers=0,gap='empty subgroup',methods={},comparisons={});continue
        samples[str(group)]=s;row=dict(controllers=int(mask.sum()),types=len(set(types[mask])),methods={},comparisons={},events={})
        for mi,m in enumerate(METHODS):
            row['methods'][m]=dict(**summary(s[:,mi]),unknown_upper=summary(s[:,8+mi]));event={k:100*macro(d[k][:,:,mi].mean((1,2)),types,mask) for k in ('collision','timeout','unknown','triggered')};num=macro(np.where(d['trigger_steps'][:,:,mi]>=0,d['trigger_steps'][:,:,mi],0).mean((1,2)),types,mask);den=event['triggered']/100;event['mean_trigger_step_when_triggered']=num/den if den else None;row['events'][m]=event
        for pi,((a,b),key) in enumerate(zip(PAIRS,PAIR_NAMES)):
            ll=s[:,16+2*pi];uu=s[:,17+2*pi];row['comparisons'][key]=dict(method_a=METHODS[a],method_b=METHODS[b],**summary(s[:,a]-s[:,b]),unknown_bounds=[float(ll[0]*100),float(uu[0]*100)],unknown_lower_CI95=np.quantile(ll[1:]*100,[.025,.975]),unknown_upper_CI95=np.quantile(uu[1:]*100,[.025,.975]),same_UID_unknown_cancelled=True)
        row['harm_online_vs_safety_ge10pp']=int(np.sum(mask&(safety-online>=.1-1e-12)));row['harm_online_vs_first_ge10pp']=int(np.sum(mask&(first-online>=.1-1e-12)));results[str(group)]=row
    high=results['只加安全投影 ≥ 98%'];low=results['只加安全投影 < 90%'];T2=dict(high_available=high['controllers']>0,high_gain=high['comparisons'].get('T2_online'),high_harm_vs_safety=high.get('harm_online_vs_safety_ge10pp'),low_available=low['controllers']>0,low_gain=low['comparisons'].get('T2_online'));T2['high_goal_met']=bool(high['controllers'] and high['comparisons']['T2_online']['point']>=0 and high['harm_online_vs_safety_ge10pp']==0);T2['low_goal_met']=bool(low['controllers'] and low['comparisons']['T2_online']['CI95'][0]>-1)
    T6={full_type(t):{m:dict(**results[t]['methods'][m],point_at_least90=bool(results[t]['methods'][m]['point']>=90)) for m in (METHODS[1],METHODS[2])} for t in sorted(LEARNING)}
    result=dict(utc=utc(),evidence='DEVELOPMENT',bootstrap_draws=5000,bootstrap_seed=2026102064,groups=results,T2=T2,T6=T6,freeze_prerequisite={m:all(r[m]['point_at_least90'] for r in T6.values()) for m in (METHODS[1],METHODS[2])},selected_threshold_counts=theta,final_three_unseen_types_not_tested=True)
    write(HERE/'T1_T7_RESULTS.json',result);np.savez_compressed(HERE/'T1_T7_BOOTSTRAP.npz',**samples)
    with (HERE/'AXIS_TABLE.csv').open('w') as f:
        fields=['类型 / 分组',*METHODS,'证据等级'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for group in [*sorted(set(types)),*['类型等权合并','学习型','优化型','搜索型','只加安全投影 ≥ 98%','90–98%','只加安全投影 < 90%']]:
            rr=results[group];out={'类型 / 分组':full_type(group),'证据等级':'DEVELOPMENT'}
            for m in METHODS:
                r=rr['methods'].get(m);out[m]='N/A' if r is None else f"{r['point']:.2f} [{r['CI95'][0]:.2f}, {r['CI95'][1]:.2f}]"
            writer.writerow(out)
    with (HERE/'CONTROLLER_RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['controller','type',*METHODS,'harm_online_vs_safety_ge10pp']);writer.writeheader()
        for ci,n in enumerate(names):writer.writerow(dict(controller=str(n),type=full_type(types[ci]),**{m:float(lower[ci,:,mi].mean()*100) for mi,m in enumerate(METHODS)},harm_online_vs_safety_ge10pp=int(safety[ci]-online[ci]>=.1-1e-12)))
    write(HERE/'EVENT_AND_CHOICES.json',dict(events={k:r.get('events') for k,r in results.items()},selected_threshold_counts=theta,trigger_steps=d['trigger_steps'].tolist(),names=names,types=types))
    from oracle_support_c6 import main as support
    support()
    from failure_diagnosis_c6 import main as failures
    failures()
    seal('T1_T7_STATISTICS.receipt.json',[HERE/n for n in ('T1_T7_RESULTS.json','T1_T7_BOOTSTRAP.npz','AXIS_TABLE.csv','CONTROLLER_RESULTS.csv','EVENT_AND_CHOICES.json','GROUP_MEMBERSHIP.json','TYPE_INFORMATION_SUPPORT.receipt.json','FAILURE_DIAGNOSIS.receipt.json','STATISTICAL_DEFINITIONS.receipt.json','statistics_c6.py','NEW_EVAL_BUNDLE.receipt.json')],numeric_only=True,no_plotting_imports=True)

if __name__=='__main__':main()
