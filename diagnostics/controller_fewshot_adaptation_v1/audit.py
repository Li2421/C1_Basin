from core import *
from scipy.special import betaln,logsumexp,expit

def main():
    gate();freeze=read(HERE/'ADAPTATION_FREEZE.json');assert freeze['preregistration_sha256']==sha(HERE/'PREREGISTRATION.md');changed=[x['path'] for x in freeze['files'] if sha(x['path'])!=x['sha256']];assert not changed,changed
    records=[];source=read(HERE/'PROTOCOL.json');weights_error=0.;offset_error=0.
    for target in range(14):
        z=np.load(HERE/'simulations'/f'{target}.npz');pool=np.load(HERE/'pools'/f'{target}.npz');p=z['paths'];ys=z['sampled_outcomes'];assert np.array_equal(ys,pool['outcomes'][p]);assert all(len(np.unique(row))==128 for row in p.reshape(-1,128));assert not z['unable_to_collect'].any()
        assert set(z['source_target_indices']).isdisjoint([target]) and len(z['source_target_indices'])==(6 if target<7 else 13)
        assert np.isin(pool['cells'][p[1],1],[2,4,10,13]).all()
        for bi,B in enumerate(BUDGETS):
            counts=z['success_counts'][:,:,bi]+z['failure_counts'][:,:,bi];valid=(ys[:,:,:int(B)]>=0).sum(-1);assert np.array_equal(counts.sum(-1),valid)
            sa=z['source_alpha'];sb=z['source_beta'];s=z['success_counts'][:,:,bi];f=z['failure_counts'][:,:,bi];lw=(betaln(sa[None,None]+s[:,:,None],sb[None,None]+f[:,:,None])-betaln(sa,sb)[None,None]).sum(-1);w=np.exp(lw-logsumexp(lw,axis=-1,keepdims=True));error=float(np.max(abs(w-z['H_component_weights'][:,:,bi])));weights_error=max(weights_error,error);assert error<1e-8
        if target<7:
            pred=np.load(HERE/'predictions'/f'{target}.npz');qp=expit(np.stack([pred['pool_R0'],pred['pool_F2']]))
            for st,rep,B in [(0,0,4),(1,17,32),(2,199,128)]:
                bi=int(np.flatnonzero(BUDGETS==B)[0]);ids=p[st,rep,:B];cells=pool['cells'][ids];labels=pool['outcomes'][ids];res=np.zeros((2,3,16));n=np.zeros(16)
                for cell,y in zip(cells,labels):
                    if y<0:continue
                    state,eta,seed=cell;res[:,:,eta]+=y-qp[:,:,state,eta];n[eta]+=1
                error=float(np.max(abs(res/(n+8)-z['offsets'][st,rep,bi])));offset_error=max(offset_error,error);assert error<1e-7
        records.append(dict(target=target,controller=source['controller_order'][target],paths_without_replacement=True,unknown_consumes_budget_without_Beta_or_offset_update=True,source_only=True,full_pool_identity_preserved=True))
    r0=read(HERE/'endpoints/main_B0.json');r7=read(R7/'real_results.json');baseline_checks=[]
    for row in r0['rows']:
        if row['key'] not in ('U|C-F2-safe','U|C-R0'):continue
        policy='conservative' if row['key']=='U|C-F2-safe' else 'argmax';rep='R3-PCA2' if policy=='conservative' else 'B0';old=next(x for x in r7['rows'] if x['coverage']=='REAL6' and x['controller']==row['controller'] and x['representation']==rep and x['policy']==policy);err=max(abs(row[k]-old[oldk]) for k,oldk in [('success_lower','success_lower'),('success_upper','success_upper'),('gain_lower','gain_src_lower'),('gain_upper','gain_src_upper')]);assert err<1e-10,(row['key'],row['controller'],err);baseline_checks.append(dict(key=row['key'],controller=row['controller'],error=err))
    # Independent bilinear scalar check of exact crossed resampling.
    endpoint=np.load(R4/'LABELS.npz')['outcomes'][1,128:,:,4:8];y=(endpoint==1).mean(-1);u=(endpoint==-1).mean(-1);sim=np.load(HERE/'simulations/1.npz');a=sim['choices'][1,2,METHODS.index('E-safe')];b=np.full_like(a,10);lo,hi=contrast(y[None,None],u[None,None],a,b);vectors=np.stack([lo.mean(1),hi.mean(1)]);bp=np.load(HERE/'endpoints/main_B8_bootstrap.npz');cols=json.loads(str(bp['columns_json']));ci=next(i for i,c in enumerate(cols) if c['key']=='R|E-safe');weights=np.load(HERE/'bootstrap_main.npz')
    for draw in (0,17,4999):
        direct=np.array([np.sum(v*weights['repeat_weights'][1,draw,:,None]*weights['state_weights'][draw,None,:]) for v in vectors]);assert np.max(abs(direct-bp['target_bootstrap'][1,draw,ci]))<1e-10
    old=read(HERE/'HISTORICAL_ASSET_INDEX.json');historical_changed=[x['path'] for x in old['files'] if sha(x['path'])!=x['sha256']];assert not historical_changed,historical_changed
    result=dict(utc=utc(),status='PASS',new_rollouts=0,new_probes=0,shared_db_writes=0,database_connections=0,preregistration_unchanged=True,adaptation_freeze_unchanged=True,all_target_sampling_checks=records,H_conjugate_weight_max_error=weights_error,offset_probability_space_max_error=offset_error,zero_budget_R0_and_F2_exact_reuse_checks=baseline_checks,exact_crossed_bootstrap_scalar_check=True,posterior_math=read(HERE/'POSTERIOR_MATH_CHECK.json'),historical_files_checked=len(old['files']),historical_changed=historical_changed,main_endpoint_seeds=[4,5,6,7],hindsight_seeds=[0,1,2,3],stageB_descriptive_only=True,all_zero_rollout_budget_data_are_existing_requests=True,source_knowledge_arms_are_specific_implementation_not_universal_claim=True)
    write(HERE/'FINAL_AUDIT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('all_target_sampling_checks','zero_budget_R0_and_F2_exact_reuse_checks','posterior_math')}))
if __name__=='__main__':main()
