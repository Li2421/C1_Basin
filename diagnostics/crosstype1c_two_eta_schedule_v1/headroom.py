"""Pre-critic proposal headroom, stage complementarity and distances."""
from common import *
def average(values,types,te=True):
    per=np.asarray(values).mean(tuple(range(1,np.asarray(values).ndim)))
    return float(np.mean([per[types==t].mean() for t in sorted(set(types))])) if te else float(per.mean())
def bootstrap_delta(lo,hi,types):
    rng=np.random.default_rng(202610190201);draw=[];groups=sorted(set(types));ns=lo.shape[1]
    for _ in range(5000):
        vals=[]
        for t in rng.choice(groups,len(groups),replace=True):
            pool=np.flatnonzero(types==t);ii=rng.choice(pool,len(pool),replace=True);ss=rng.integers(ns,size=(len(ii),ns));vals.append(np.asarray([lo[ii[:,None],ss].mean(),hi[ii[:,None],ss].mean()]))
        draw.append(np.mean(vals,0))
    return [float(np.quantile(np.asarray(draw)[:,0],.025)*100),float(np.quantile(np.asarray(draw)[:,1],.975)*100)]
def main(which):
    target='PILOT_TRAINING_BUNDLE' if which=='pilot' else 'TRAINING_BUNDLE'
    with np.load(HERE/f'{target}.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(HERE/('PILOT_TRAINING_REFERENCES.npz' if which=='pilot' else 'TRAINING_REFERENCES.npz'),allow_pickle=False) as z:references={k:z[k] for k in z.files}
    labels=data['labels'][...,0];lo=(labels==1).astype(float);hi=(labels!=0).astype(float);types=data['types'];diag=lo[:,:,:8].max(-1);off=lo[:,:,8:].max(-1);dl=diag-hi[:,:,8:].max(-1);du=hi[:,:,:8].max(-1)-off
    summaries={}
    for group in [*sorted(set(types)),'combined']:
        mask=np.ones(len(types),bool) if group=='combined' else types==group;tt=types[mask]
        summaries[str(group)]={}
        for te in ([True,False] if group=='combined' else [True]):
            summaries[str(group)]['type_equal' if te else 'controller_equal']=dict(diagonal_oracle_pct=100*average(diag[mask],tt,te),non_diagonal_oracle_pct=100*average(off[mask],tt,te),diagonal_minus_non_diagonal_pp=100*average((diag-off)[mask],tt,te),unknown_bounds_pp=[100*average(dl[mask],tt,te),100*average(du[mask],tt,te)],CI95_pp=bootstrap_delta(dl[mask],du[mask],tt) if te else None,human_step500_reference_pct=100*average(references['labels'][mask,:,1,0]==1,tt,te),evidence='SAMPLED_ACTIONS_IN_SAMPLE_SEED0')
    traces=read(HERE/f'{target}_TRACE_INDEX.json');records=[];cache={}
    def progress(rid,t):
        if rid not in traces:return None
        entry=traces[rid]
        if rid not in cache:
            assert sha(entry['trace_path'])==entry['trace_sha256']
            with np.load(entry['trace_path'],allow_pickle=False) as z:p=z['positions'];goals=z['goals']
            distance=np.linalg.norm(goals[None]-p,axis=-1).sum(-1);cache[rid]=(distance[0]-distance)/distance[0]
        values=cache[rid]
        return float(values[t]) if len(values)>t else None
    for i,name in enumerate(data['names']):
        for s in range(48):
            for a in range(8,26):
                cid=int(data['candidate_indices'][i,s,a]);pair=cid//5;e,l=pair//8,pair%8;tim=(200,300,400,500,600)[cid%5];ep=progress(str(data['uids'][i,s,e,0]),200);lp=progress(str(data['uids'][i,s,l,0]),200);known=labels[i,s,l]>=0
                actual_e=progress(str(data['uids'][i,s,e,0]),tim);actual_l=progress(str(data['uids'][i,s,l,0]),tim)
                records.append(dict(controller=str(name),type=str(types[i]),state=s,candidate=cid,early_index=e,late_index=l,switch_step=tim,supported_200=ep is not None and lp is not None and known,complementary_200=(ep>lp and labels[i,s,l]==1) if ep is not None and lp is not None and known else None,supported_at_switch=actual_e is not None and actual_l is not None and known,complementary_at_switch=(actual_e>actual_l and labels[i,s,l]==1) if actual_e is not None and actual_l is not None and known else None))
    complement={}
    for kind in sorted(set(types)):
        rr=[r for r in records if r['type']==kind];support=[r for r in rr if r['supported_200']];actual=[r for r in rr if r['supported_at_switch']]
        complement[str(kind)]=dict(total_drawn_schedules=len(rr),supported_at200=len(support),fraction_at200=float(np.mean([r['complementary_200'] for r in support])) if support else None,supported_at_actual_switch=len(actual),fraction_at_actual_switch=float(np.mean([r['complementary_at_switch'] for r in actual])) if actual else None,limitation='Conditional on actually observed fixed-action traces and known late outcome; no post-termination trajectory imputation or causal claim')
    base=read(HERE/'BASE_PROTOCOL.json');proposals=data['proposals'];distances={str(k):dict(all_proposal_distance_quantiles=np.quantile(np.linalg.norm(proposals-np.asarray(base['eta'][k]),axis=-1),[0,.25,.5,.75,1]).tolist(),nearest_per_state_quantiles=np.quantile(np.linalg.norm(proposals-np.asarray(base['eta'][k]),axis=-1).min(-1),[0,.25,.5,.75,1]).tolist()) for k in (13,10)}
    name='PILOT_REPORT' if which=='pilot' else 'SELECTABLE_SPACE';write(HERE/f'{name}.json',dict(utc=utc(),evidence='DEVELOPMENT',model_used=False,summaries=summaries,stage_complementarity=complement,proposal_distances_to_human_references=distances,training_crossfit='Unavailable: only seed0 for pure generated schedule labels',pilot_does_not_change_subsequent_design=True));write(HERE/f'{name}_COMPLEMENTARITY_DETAILS.json',records)
    (HERE/f'{name}.md').write_text('# 开发前提议集核查\n\n'+json.dumps(read(HERE/f'{name}.json'),ensure_ascii=False,indent=2)+'\n')
    seal(f'{name}.receipt.json',[HERE/f'{name}.json',HERE/f'{name}.md',HERE/f'{name}_COMPLEMENTARITY_DETAILS.json',HERE/f'{target}.receipt.json',HERE/'headroom.py'],before_critic_training=True)
    print(name,'complete',summaries['combined']['type_equal'],flush=True)
if __name__=='__main__':main(sys.argv[1])
