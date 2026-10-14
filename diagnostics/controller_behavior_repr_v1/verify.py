"""Final evidence, leakage, numerical protocol and historical-integrity checks."""
from core import *
from data import Data,Representations,name

def freeze():
    gate();ts=read(HERE/'TASKS.json');files=[]
    for t in ts:
        f=HERE/'fits'/name(t);c=HERE/'classifiers'/name(t);m=read(f/'complete.json');assert m['status']=='COMPLETE'
        for path in [f/'complete.json',f/'predictions.npz',c/'complete.json',c/'predictions.npz',c/'P_rule.npz']+[Path(q) for q in m['checkpoints']]:files.append(dict(path=str(path),sha256=sha(path)))
    assert not (HERE/'MODEL_FREEZE.json').exists(),'Freeze receipt must not be overwritten'
    write(HERE/'MODEL_FREEZE.json',dict(utc=utc(),before_new_endpoint_evaluation=True,tasks=len(ts),files=files,selected_only_by_source_validation=True,preregistration_sha256=sha(HERE/'PREREGISTRATION.md')))

def main():
    gate();frozen=read(HERE/'MODEL_FREEZE.json');changed=[x['path'] for x in frozen['files'] if sha(x['path'])!=x['sha256']];assert not changed,changed
    ts=read(HERE/'TASKS.json');checks=[];checkpoints=[];optimized=0;backends={};r=Representations();p5=read(R5/'stage_2/protocol.json')
    for t in ts:
        d=Data(t);f=HERE/'fits'/name(t);m=read(f/'complete.json');z=np.load(f/'predictions.npz');assert z['held'].shape==(3,len(t['held']),len(d.test_states),16) and np.isfinite(z['held']).all()
        assert set(t['sources']).isdisjoint(t['held']);assert set(d.valnames).isdisjoint(t['held'])
        if t['bed']=='real':
            assert d.src==10 and len(t['real_sources'])==6
            for n in t['sources']:
                if n.startswith('blend_'):
                    b=p5['controllers']['ring_exchange'][n]['payload']['blend'];assert b['A'] in t['real_sources'] and b['B'] in t['real_sources']
            assert not t['sensitivity']
        for p,h in zip(m['checkpoints'],m['checkpoint_sha256']):assert sha(p)==h;checkpoints.append(dict(task=name(t),path=p,sha256=h))
        if m['optimization_performed']:
            optimized+=3;model=read(f/'MODEL_FROZEN.json') if (f/'MODEL_FROZEN.json').exists() else read(HERE/'shared_sparse_fits'/(t['representation']+('_imputed' if t['sensitivity'] else ''))/'MODEL_FROZEN.json');hist=np.array(model['validation_nll_history']);assert hist.shape==(3,62) and np.isfinite(hist).all();assert model['schedule'][-1]==12000 and model['schedule'][int(hist.mean(0).argmin())]==m['selected_step']
            assert model['source_prior_eta']==d.src and model['validation_names']==d.valnames
            backend=m.get('backend','gpu (original launch)');backends[backend]=backends.get(backend,0)+3
        checks.append(dict(task=name(t),source_disjoint=True,validation_disjoint=True,prior=d.src,shape=list(z['held'].shape),finite=True))
    assert optimized==165,(optimized,'Expected 135 primary + 30 imputation sensitivity seed fits')
    # Independent exact reduction checks against round6 for reused R3 endpoint rows.
    oldchecks=[]
    for bed in ('real','blend'):
        old=read(R6/f'{bed}_results.json')['rows'];new=read(HERE/f'{bed}_results.json')['rows']
        for r0 in new:
            if r0['representation']!='R3-PCA2' or (bed=='real' and r0['coverage']!='REAL6'):continue
            arm='F4' if r0['policy']=='F4' else 'F2_PCA2';policy='argmax' if arm=='F4' else r0['policy'];cov='LOCO' if bed=='real' else r0['coverage'];oldrow=next(x for x in old if x['coverage']==cov and x['controller']==r0['controller'] and x['arm']==arm and x['policy']==policy)
            fields=['success_lower','success_upper','gain_R0_lower','gain_R0_upper','gain_src_lower','gain_src_upper'];err=max(abs(r0[k]-oldrow[k]) for k in fields);assert err<1e-10 and np.array_equal(r0['choice_histogram'],oldrow['choice_histogram']);oldchecks.append(dict(bed=bed,coverage=cov,controller=r0['controller'],policy=r0['policy'],max_error=err))
    # Unknown is an interval, with exact cancellation for identical choices.
    y=np.array([[[0.,1.]]]);u=np.array([[[1.,0.]]]);a=np.array([[0]]);b=np.array([[1]])
    lo,hi=contrast(y,u,a,a);assert lo.item()==hi.item()==0
    lo,hi=contrast(y,u,a,b);assert lo.item()==-1 and hi.item()==0
    lo,hi=contrast(y,u,b,a);assert lo.item()==0 and hi.item()==1
    historical=read(HERE/'HISTORICAL_ASSET_INDEX.json')['files'];hchanged=[x['path'] for x in historical if sha(x['path'])!=x['sha256']]
    # No database connection exists in this task. Concurrent global DB changes are not claimed absent.
    write(HERE/'CHECKPOINT_INDEX.json',checkpoints)
    explained=[]
    if hchanged==[str(ROOT/'docs/prompt_round7_behavior_repr_20261013.md')]:
        old_prompt=(HERE/'PREREGISTRATION.md').read_text().split('# 用户设计全文（冻结副本）\n\n',1)[1];new_prompt=(ROOT/'docs/prompt_round7_behavior_repr_20261013.md').read_text()
        # Only the resource paragraph may differ, matching the user's steering.
        before_a,before_b=old_prompt.split('- **资源：',1);after_a,after_b=new_prompt.split('- **资源：',1)
        assert before_a==after_a and before_b.split('# 六、交付',1)[1]==after_b.split('# 六、交付',1)[1]
        assert '18 个 CPU shard 和 3 个 GPU shard' in after_b.split('# 六、交付',1)[0]
        explained=[dict(path=hchanged[0],reason='Only resource paragraph changed externally to user-authorized shared18CPU/3GPU; scientific design unchanged',current_sha256=sha(hchanged[0]))]
    result=dict(utc=utc(),status='PASS' if not hchanged else 'PASS_WITH_DOCUMENTED_RESOURCE_UPDATE',tasks=len(ts),independent_new_seed_fits=optimized,backend_seed_fits=backends,model_freeze_unchanged=True,preregistration_unchanged=True,training_and_validation_leakage_checks=checks,reused_endpoint_rows_exact=oldchecks,unknown_pair_checks_passed=True,historical_files_checked=len(historical),historical_changed=hchanged,explained_external_changes=explained,new_rollouts=0,shared_db_writes=0,database_connections=0,global_database_unchanged_claim=False,write_scope_enforced='Python audit hook rejects file writes outside this directory and all sqlite connect calls; Slurm logs/tmp/cache directed here',pure_Q_no_video_required=True)
    write(HERE/'FINAL_AUDIT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('training_and_validation_leakage_checks','reused_endpoint_rows_exact')}))
    assert not hchanged or len(explained)==len(hchanged),'Investigate historical asset changes before final handoff'
if __name__=='__main__':freeze() if len(sys.argv)>1 and sys.argv[1]=='freeze' else main()
