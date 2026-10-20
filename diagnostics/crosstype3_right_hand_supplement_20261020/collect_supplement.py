from common import *
import csv,subprocess
SRC=D/'crosstype3_unseen_controller_schedule_v1'
TYPES=('transformer_bc','gradient_mpc','conflict_based_search')
METHODS=('right_hand','all_safety_baseline','self_schedule','no_controller')
LABELS={'right_hand':'固定参数持续右手规则＋安全投影','all_safety_baseline':'仅安全投影','self_schedule':'学习双η＋学习切换','no_controller':'学习双η＋学习切换·NoC'}
def main():
    assert (HERE/'CPU_RESTORED.json').exists(),'Restore borrowed CPU first'
    rows=read(HERE/'evaluation_CASES.json');todo=read(HERE/'evaluation_MISSING_CASES.json')
    for c in todo:assert (HERE/'evaluation/raw'/f"{c['rollout_uid']}.json").exists(),c['rollout_uid']
    # One official merger, after all supplement writers have finished.
    result=subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,text=True,capture_output=True)
    (HERE/'MERGE.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
    rhr={(c['base_controller_name'],c['state_uid'],c['seed']):c for c in rows}
    idx=[r for r in read(SRC/'NEW_EVAL_METHOD_UIDS.json') if r['cohort']=='unseen'];assert len(idx)==864
    details=[];arrays={};rng=np.random.default_rng(202610200601);boots={};totals={}
    with rollout_db.connect(True) as db:
        def result_for(rid):
            r=db.execute('SELECT * FROM rollout WHERE rollout_uid=? AND conflict_quarantined=0',(rid,)).fetchone()
            if r is not None:return dict(r)
            p=SRC/'new_evaluation/raw'/f'{rid}.json';assert p.exists(),rid
            return read(p)
        for kind in TYPES:
            controllers=sorted({r['controller'] for r in idx if r['type']==kind});assert len(controllers)==6
            stateids=sorted({r['state_uid'] for r in idx if r['type']==kind});assert len(stateids)==48
            ci={n:i for i,n in enumerate(controllers)};si={s:i for i,s in enumerate(stateids)}
            lo=np.zeros((4,6,48,2));up=np.zeros_like(lo);coll=np.zeros_like(lo);timeout=np.zeros_like(lo);unknown=np.zeros_like(lo)
            for r in idx:
                if r['type']!=kind:continue
                for m,method in enumerate(METHODS):
                    uids=[rhr[(r['controller'],r['state_uid'],q)]['rollout_uid'] for q in (0,1)] if method=='right_hand' else r['method_uids'][method]
                    for q,rid in enumerate(uids):
                        v=result_for(rid);u=bool(v['numerical_failure']);success=bool(v['success']) and not u
                        ix=(m,ci[r['controller']],si[r['state_uid']],q);lo[ix]=success;up[ix]=success or u;coll[ix]=v['collision'];timeout[ix]=v['timeout'];unknown[ix]=u
                        details.append(dict(type=kind,controller=r['controller'],state_uid=r['state_uid'],seed=q,method=method,rollout_uid=rid,success_lower=int(success),success_upper=int(success or u),collision=v['collision'],timeout=v['timeout'],unknown=int(u)))
            cs=rng.integers(6,size=(5000,6));ss=rng.integers(48,size=(5000,6,48))
            blo=lo[:,cs[:,:,None],ss,:].mean(axis=(2,3,4))*100
            bup=up[:,cs[:,:,None],ss,:].mean(axis=(2,3,4))*100
            # Advanced indexing above gives method x bootstrap x controller x state x seed.
            assert blo.shape==(4,5000),blo.shape
            boots[kind]=(blo,bup);methods={}
            for m,method in enumerate(METHODS):
                methods[method]=dict(success_lower_pct=float(lo[m].mean()*100),success_upper_pct=float(up[m].mean()*100),CI95_pct=[float(np.quantile(blo[m],.025)),float(np.quantile(bup[m],.975))],success_count=int(lo[m].sum()),unknown_count=int(unknown[m].sum()),collision_count=int(coll[m].sum()),timeout_count=int(timeout[m].sum()),episodes=576)
            deltas={}
            for m in (1,2,3):deltas[METHODS[m]]=dict(lower_pp=float((lo[0]-up[m]).mean()*100),upper_pp=float((up[0]-lo[m]).mean()*100),CI95_pp=[float(np.quantile(blo[0]-bup[m],.025)),float(np.quantile(bup[0]-blo[m],.975))])
            totals[kind]=dict(methods=methods,right_hand_minus=deltas)
            arrays[kind]=(lo,up,coll,timeout,unknown,controllers)
    combined={}
    for m,method in enumerate(METHODS):
        bl=np.mean([boots[t][0][m] for t in TYPES],axis=0);bu=np.mean([boots[t][1][m] for t in TYPES],axis=0)
        combined[method]=dict(success_lower_pct=float(np.mean([totals[t]['methods'][method]['success_lower_pct'] for t in TYPES])),success_upper_pct=float(np.mean([totals[t]['methods'][method]['success_upper_pct'] for t in TYPES])),CI95_pct=[float(np.quantile(bl,.025)),float(np.quantile(bu,.975))])
    comparison={}
    for m in (1,2,3):
        bl=np.mean([boots[t][0][0]-boots[t][1][m] for t in TYPES],axis=0);bu=np.mean([boots[t][1][0]-boots[t][0][m] for t in TYPES],axis=0)
        comparison[METHODS[m]]=dict(lower_pp=float(np.mean([totals[t]['right_hand_minus'][METHODS[m]]['lower_pp'] for t in TYPES])),upper_pp=float(np.mean([totals[t]['right_hand_minus'][METHODS[m]]['upper_pp'] for t in TYPES])),CI95_pp=[float(np.quantile(bl,.025)),float(np.quantile(bu,.975))])
    with (HERE/'EPISODE_RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    records=[]
    for kind,(lo,up,coll,timeout,unknown,controllers) in arrays.items():
        for i,name in enumerate(controllers):
            records.append(dict(type=kind,controller=name,episodes=96,success_count=int(lo[0,i].sum()),success_lower_pct=float(lo[0,i].mean()*100),success_upper_pct=float(up[0,i].mean()*100),collision_count=int(coll[0,i].sum()),timeout_count=int(timeout[0,i].sum()),unknown_count=int(unknown[0,i].sum())))
    with (HERE/'CONTROLLER_RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    hashes=read(HERE/'SOURCE_COPY_HASHES.json');hashes.update(read(HERE/'INPUT_HASHES.json'));changed=[p for p,h in hashes.items() if sha(p)!=h]
    write(HERE/'INTEGRITY_CHECK.json',dict(utc=utc(),checked=len(hashes),changed=changed,passed=not changed));assert not changed,changed
    post=planner.preflight(HERE/'evaluation_PLANNED_ROLLOUTS.json');write(HERE/'evaluation_POSTFLIGHT.json',post)
    with rollout_db.connect(True) as db:
        present=sum(db.execute('SELECT 1 FROM rollout WHERE rollout_uid=? AND conflict_quarantined=0',(c['rollout_uid'],)).fetchone() is not None for c in rows)
    assert present==1728,present
    write(HERE/'RESULTS.json',dict(utc=utc(),types=totals,type_equal=combined,right_hand_minus_type_equal=comparison,physical_requests=1728,DB_present=present,bootstrap=dict(draws=5000,seed=202610200601,method='fixed three types; resample controllers then states within each sampled controller; two seeds remain clustered; unknown lower/upper envelope'),CPU_restored=read(HERE/'CPU_RESTORED.json')))
    lines=['# 三个原未见类型的右手规则补评','', '固定历史右手规则：40步进度窗口、0.05进度阈值、0.35切向速度；最终经过原硬安全投影。无训练和参数选择，复用跨类型3的18个controller、48个state和两个seed；1728请求均为精确缓存缺失。此为打开类型结果后的补充对照，不重新声称盲化确认。','', '| 类型 | 右手规则＋安全投影 | 仅安全投影 | 学习双η＋切换 | 学习双η＋切换·NoC |','|---|---:|---:|---:|---:|']
    for kind in (*TYPES,'type_equal'):
        vals=combined if kind=='type_equal' else totals[kind]['methods'];columns=[]
        for method in METHODS:
            v=vals[method];columns.append(f"{v['success_lower_pct']:.2f}% [{v['CI95_pct'][0]:.2f}, {v['CI95_pct'][1]:.2f}]")
        lines.append('| '+kind+' | '+' | '.join(columns)+' |')
    lines+=['','成功率点估计使用unknown下界，区间包络unknown上下界。所有比较按相同controller/state/seed配对；逐条身份见EPISODE_RESULTS.csv。超时只报告超时，不作为死锁判定。','', '历史RHR及三个新类型Safety复放的位置/速度/动作逐位一致。旧源码与输入哈希检查通过。18CPU已归还跨类型6，后者保留已完成/unknown记录，从未完成案例续跑。恢复收据见CPU_RESTORED.json。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(types={t:totals[t]['methods']['right_hand'] for t in TYPES},type_equal=combined['right_hand'])),flush=True)
if __name__=='__main__':main()
