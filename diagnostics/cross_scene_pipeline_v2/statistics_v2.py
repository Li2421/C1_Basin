"""Prespecified paired hierarchical development statistics; immutable choices."""
from common import *
from names_v2 import METHOD_NAMES
from statistics_new import estimate
from diagnose_v1 import ranking
from results import result
from anchored import sigmoid
import numpy as np
from scipy.stats import spearmanr

SEED=2026102517
CORE_METHODS=tuple(m for m in METHOD_NAMES if m!='generator_only')
PAIRS={'v2_minus_global_fixed':('v2','global_fixed'),'v2_minus_eta_A':('v2','eta_A'),'v2_minus_knn':('v2','knn'),
 'v2_minus_v1_pipeline':('v2','v1_pipeline'),'v2_minus_v2_49':('v2','v2_49')}
ATTRIBUTIONS={f'v2_49_minus_{m}':('v2_49',m)for m in('v2_49_raw_description','v2_49_no_fallback','v2_49_clip','v2_49_no_prior','v2_49_lambda1','v2_49_per_cell','v2_49_per_controller')}

def assemble():
    old=read_evidence(ROOT/'diagnostics/cross_scene_correction_v1/LABEL_TABLES.json');v1=read_evidence(ROOT/'diagnostics/cross_scene_pipeline_v1/EVALUATION_LABEL_TABLES.json')
    refs={(r['fold'],r['scene'],r['controller'],r['state_index'],r['method']):r for r in read(HERE/'main_REFERENCES.json')};out={};missing=[]
    for fold in read(HERE/'FOLDS.json'):
        group=fold['kind']if fold['kind']!='lofo'else fold['name'];target=out.setdefault(group,{})
        for name,ch in read(HERE/'folds'/fold['name']/'CHOICES.json').items():
            source=old[name];sy=np.asarray(source['y'],float);su=np.asarray(source['unknown'],float);C,S,A=sy.shape
            assert ch['controllers']==source['controllers']and ch['state_indices']==source['state_indices']
            methods=tuple(METHOD_NAMES)if fold['generator']else('v2_49','global_fixed');y=np.zeros((C,S,len(methods)));u=np.zeros_like(y);done=np.ones_like(y,bool);applicable=np.ones_like(y,bool)
            for mi,method in enumerate(methods):
                choice=ch['methods'][method]
                if not choice['applicable']:applicable[:,:,mi]=False;done[:,:,mi]=False;continue
                if method=='v1_pipeline':
                    previous=v1[group][name];index=previous['methods'].index('pipeline');assert previous['controllers']==ch['controllers']and previous['state_indices']==ch['state_indices']
                    y[:,:,mi]=np.asarray(previous['y'])[:,:,index];u[:,:,mi]=np.asarray(previous['unknown'])[:,:,index]
                elif method=='scene_xstate':
                    weights=np.asarray(choice['weights_by_state']);y[:,:,mi]=(sy*weights[None]).sum(-1);u[:,:,mi]=(su*weights[None]).sum(-1)
                elif method in('v2','generator_only'):
                    for ci,ctl in enumerate(ch['controllers']):
                        for si,st in enumerate(ch['state_indices']):
                            ref=refs[(fold['name'],name,ctl,st,method)]
                            if ref['source']=='historical_49_label':
                                a=ref['historical_action'];y[ci,si,mi]=sy[ci,si,a];u[ci,si,mi]=su[ci,si,a]
                            else:
                                try:r=result('main',ref['rollout_uid'])
                                except FileNotFoundError:
                                    done[ci,si,mi]=False;missing.append(ref);continue
                                y[ci,si,mi]=int(r['success']);u[ci,si,mi]=int(r['numerical_failure'])
                else:
                    selected=np.asarray(choice['indices'],int);assert np.all(selected<49)
                    y[:,:,mi]=np.take_along_axis(sy,selected[:,:,None],2)[:,:,0];u[:,:,mi]=np.take_along_axis(su,selected[:,:,None],2)[:,:,0]
            target[name]=dict(fold=fold['name'],controllers=ch['controllers'],state_indices=ch['state_indices'],methods=methods,y=y.tolist(),unknown=u.tolist(),
                completed=done.tolist(),applicable=applicable.tolist(),same_pool_global_candidate=ch['methods']['global_fixed']['selected_candidates'][0]['key'],strong_global_candidate='sE29')
    write(HERE/'EVALUATION_LABEL_TABLES.json',out);write(HERE/'UNCOMPLETED_CELLS.json',dict(count=len(missing),cells=missing,main_value='Failure; completed-only results reported separately.'))
    return out

def for_methods(tables,methods):
    out={}
    for n,row in tables.items():
        ix=[row['methods'].index(m)for m in methods];r=dict(row);r['methods']=methods
        for k in('y','unknown','completed','applicable'):r[k]=np.asarray(row[k])[:,:,ix].tolist()
        out[n]=r
    return out

def completed_only(tables,methods):
    result={};scene_rows={};counts={}
    for n,row in tables.items():
        y=np.asarray(row['y']);done=np.asarray(row['completed']);app=np.asarray(row['applicable']);rates={};total={}
        for method in methods:
            i=row['methods'].index(method);valid=done[:,:,i]&app[:,:,i];values=[y[c,:,i][valid[c]].mean()for c in range(len(y))if valid[c].any()]
            rates[method]=float(np.mean(values))if values else None;total[method]=dict(completed=int(valid.sum()),applicable=int(app[:,:,i].sum()))
        scene_rows[n]=rates;counts[n]=total
    for m in methods:
        values=[row[m]for row in scene_rows.values()if row[m]is not None];result[m]=float(np.mean(values))if values else None
    comparisons={}
    for name,(a,b)in PAIRS.items():
        vals=[]
        for n,row in tables.items():
            y=np.asarray(row['y']);d=np.asarray(row['completed']);ap=np.asarray(row['applicable']);i=row['methods'].index(a);j=row['methods'].index(b);mask=d[:,:,i]&d[:,:,j]&ap[:,:,i]&ap[:,:,j]
            within=[(y[c,:,i]-y[c,:,j])[mask[c]].mean()for c in range(len(y))if mask[c].any()]
            if within:vals.append(float(np.mean(within)))
        comparisons[name]=dict(estimate=float(np.mean(vals))if vals else None,scenes_with_completed_pairs=len(vals))
    return dict(rates=result,per_scene=scene_rows,counts=counts,paired_comparisons=comparisons,scope='Only completed applicable cells, equal controller-type weights then equal scene weights; missing types/scenes omitted and explicitly counted.')

def ranking_diagnostics():
    old=read_evidence(ROOT/'diagnostics/cross_scene_correction_v1/LABEL_TABLES.json');out={}
    for fold in read(HERE/'FOLDS.json'):
        if not fold['generator']:continue
        group='loso'if fold['kind']=='loso'else fold['name'];g=out.setdefault(group,{})
        with np.load(HERE/'folds'/fold['name']/'RANKING_SCORES.npz',allow_pickle=False)as z:
            for n in fold['targets']:
                y=np.asarray(old[n]['y'],float);truth=y.mean((0,1));rows={}
                for mode in('full_selected','full_lambda1','no_prior','raw_selected','prior'):
                    score=z[n+'__'+mode];pred=score.mean((0,1))if mode=='prior'else sigmoid(score).mean((0,1));rho=float(spearmanr(pred,truth).statistic)if np.ptp(pred)>0 and np.ptp(truth)>0 else None
                    rows[mode]=dict(cell_pair_ranking_accuracy=ranking(y,score),scene_score_spearman=rho,predicted_scene_scores=pred.tolist(),actual_scene_rates=truth.tolist())
                g[n]=rows
    summaries={group:{mode:{key:float(np.mean([r[mode][key]for r in rows.values()if r[mode][key]is not None]))for key in('cell_pair_ranking_accuracy','scene_score_spearman')}for mode in('full_selected','full_lambda1','no_prior','raw_selected','prior')}for group,rows in out.items()}
    write(HERE/'RANKING_DIAGNOSTICS.json',dict(groups=out,macro=summaries,new_rollouts=0,score_scope='State-level range fallback is applied as prescribed; scene-level fallback determines execution choice and does not overwrite the underlying model scene scores. Prior uses exact p0.'))

def main():
    frozen=HERE/'STATISTICS_V2_FROZEN.receipt.json'
    if frozen.exists():
        for p,h in read(frozen)['files'].items():assert sha(p)==h,p
    data=assemble();loso=estimate(for_methods(data['loso'],CORE_METHODS),dict(PAIRS,**ATTRIBUTIONS),SEED)
    lofo={g:estimate(for_methods(data[g],CORE_METHODS),dict(PAIRS,**ATTRIBUTIONS),SEED+i+1)for i,g in enumerate(('lofo__bottleneck','lofo__literature'))}
    full=estimate(data['full18'],{'v2_49_minus_global_fixed':('v2_49','global_fixed')},SEED+3)
    primary=loso['comparisons']['v2_minus_global_fixed'];worst=min(primary['per_scene'].values());g1=primary['estimate']>=.03 and worst>=-.10
    g2=all(v['comparisons']['v2_minus_global_fixed']['estimate']>=-.05 for v in lofo.values());g3=full['comparisons']['v2_49_minus_global_fixed']['estimate']>=0
    goals=dict(G1=dict(passed=g1,main_estimate=primary['estimate'],worst_scene_difference=worst,estimate_threshold=.03,worst_scene_threshold=-.10),
        G2=dict(passed=g2,directions={k:v['comparisons']['v2_minus_global_fixed']['estimate']for k,v in lofo.items()},threshold=-.05),
        G3=dict(passed=g3,full18_estimate=full['comparisons']['v2_49_minus_global_fixed']['estimate'],threshold=0),all_passed=g1 and g2 and g3)
    generator={};generator_rates=[];generator_upper=[]
    for n,row in data['loso'].items():
        i=row['methods'].index('generator_only');app=np.asarray(row['applicable'])[:,:,i]
        if not app.any():generator[n]=dict(applicable=False);continue
        y=np.asarray(row['y'])[:,:,i];u=np.asarray(row['unknown'])[:,:,i];generator[n]=dict(applicable=True,rate=float(y.mean()),unknown_success_upper=float(np.minimum(1,y+u).mean()));generator_rates.append(y.mean());generator_upper.append(np.minimum(1,y+u).mean())
    ranking_diagnostics()
    write(HERE/'STATISTICS.json',dict(primary=primary,development_goals=goals,loso=loso,lofo=lofo,full18=full,
        completed_only={g:completed_only(data[g],CORE_METHODS)for g in('loso','lofo__bottleneck','lofo__literature')},
        generator_only=dict(per_scene=generator,applicable_scene_count=len(generator_rates),rate=float(np.mean(generator_rates)),unknown_success_upper=float(np.mean(generator_upper)),scope='Descriptive; only scenes not out of range. Not used for any development goal.'),
        uncompleted_cells=read(HERE/'UNCOMPLETED_CELLS.json')['count'],scope='开发证据：在本修正族、本控制器集与本安全层下，对11个去重场景；去重、超范围与边距在看过v1结果后确定。',
        bootstrap='5000 paired hierarchical scene -> controller type -> state draws, seed2026102517 and fixed +1/+2/+3 framework offsets; frozen fits and choices, no refitting. Unknown and unfinished cells=failure; unknown upper and completed-only results separate.'))
    print('V2 PRIMARY',primary['estimate'],primary['ci'],'GOALS',goals,flush=True)

if __name__=='__main__':main()
