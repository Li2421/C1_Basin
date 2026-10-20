from core import *
from registration import connect
from audit_utils import source_inventory
NAMES=[t['name'] for t in read(HERE/'TEACHERS.json')];ARMS=['safety baseline','F10','F13','S300','S500'];MENU=[f'K{i}' for i in range(16)]+['S300','S500']
def collect(phase,names,arms,pipe=None):
    cases=read(HERE/f'{phase}_CASES.json');rows={r['rollout_uid']:r for r in read(HERE/f'{phase}_DB_ROWS.json')};y=np.full((len(names),32,2,len(arms)),np.nan);u=np.zeros_like(y)
    for c in cases:
        if c['base_controller_name'] not in names or c['arm'] not in arms or pipe is not None and c['display_pipeline']!=pipe:continue
        r=rows[c['rollout_uid']];ix=(names.index(c['base_controller_name']),c['state_index'],c['seed'],arms.index(c['arm']));y[ix]=r['success'] if not r['numerical_failure'] else 0;u[ix]=r['numerical_failure']
    assert np.isfinite(y).all();return y,u

def historical(teachers=False):
    path=D/('decentralized_pipeline_v1' if teachers else 'teacher_student_macflow_v1c');cs=[c for c in read(path/'main_CASES.json') if c['arm']=='S500' and c['pipeline']==('central' if teachers else 'local')];y=np.full((14,32,2,1),np.nan);u=np.zeros_like(y);manifest=[]
    with connect(True) as db:
        for c in cs:
            r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r is not None and not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE';ix=(NAMES.index(c['base_controller_name']),c['state_index'],c['seed'],0);y[ix]=r['success'] if not r['numerical_failure'] else 0;u[ix]=r['numerical_failure'];manifest.append(dict(rollout_uid=c['rollout_uid'],source=str(path/'main_CASES.json'),row_sha256=hashlib.sha256(json.dumps(dict(r),sort_keys=True).encode()).hexdigest()))
    assert len(cs)==896 and np.isfinite(y).all();write(HERE/(('TEACHER' if teachers else 'STUDENT')+'_S500_REUSE.json'),manifest);return y,u

def paired(y,u,ref=None,refu=None):
    # Arrays C,state,seed,arm; same controller/state draws for all arms and deltas.
    C=len(y);rng=np.random.default_rng(2026103032);ic=rng.integers(0,C,(5000,C));js=rng.integers(0,32,(5000,C,32))
    def draw(x):return x.mean(2)[ic[:,:,None],js,:].mean((1,2))
    b=draw(y);bu=draw(u);out=[]
    for j in range(y.shape[-1]):
        v=dict(mean=float(y[...,j].mean()),unknown=int(u[...,j].sum()),identification_interval=[float(y[...,j].mean()),float((y+u)[...,j].mean())],CI95=np.quantile(b[:,j],[.025,.975]).tolist(),CI_unknown_envelope=[float(np.quantile(b[:,j],.025)),float(np.quantile((b+bu)[:,j],.975))])
        if ref is not None:
            br=draw(ref);bru=draw(refu);delta=b[:,j]-br[:,0];lo=y[...,j]-ref[...,0]-refu[...,0];hi=y[...,j]+u[...,j]-ref[...,0];v['minus_reference']=dict(mean=float((y[...,j]-ref[...,0]).mean()),CI95=np.quantile(delta,[.025,.975]).tolist(),identification_interval=[float(lo.mean()),float(hi.mean())],CI_unknown_envelope=[float(np.quantile(delta-bru[:,0],.025)),float(np.quantile(delta+bu[:,j],.975))])
        out.append(v)
    return out

def phaseA():
    out={};det=[]
    for pipe,teacher in [('local',False),('central',True)]:
        reference,ru=historical(teacher);out[pipe]={'reference_S500':paired(reference,ru)[0],'controls':{}}
        for name in ['Z','G']:
            y,u=collect('A',[name],ARMS,pipe);rows=paired(y,u);expanded=np.broadcast_to(y,reference.shape[:-1]+(5,));expu=np.broadcast_to(u,expanded.shape);delta=paired(expanded,expu,reference,ru);entries={a:dict(v,minus_reference=delta[j]['minus_reference']) for j,(a,v) in enumerate(zip(ARMS,rows))};d=entries['S500']['minus_reference']['mean'];interpret='修正本身就能完成 Ring；学生/教师贡献不显著（按预注册点估计规则）' if d>=-.05 else '学生/教师与修正有协同：修正是在修学习策略' if d<=-.20 else '部分协同';out[pipe]['controls'][name]=dict(arms=entries,S500_interpretation=interpret if name=='Z' else '描述性比较')
    # Compare action/positions/velocities/goals and all outcome flags, ignoring seed metadata.
    cases=read(HERE/'A_CASES.json');lookup={(c['kind'],c['display_pipeline'],c['arm'],c['state_index'],c['seed']):c for c in cases};rows={r['rollout_uid']:r for r in read(HERE/'A_DB_ROWS.json')}
    mismatches=[];unknownpairs=[]
    for key,c in lookup.items():
        if c['seed']!=0:continue
        d=lookup[key[:-1]+(1,)];r,s=rows[c['rollout_uid']],rows[d['rollout_uid']];a,b=[read(HERE/'A/raw'/f"{x['rollout_uid']}.json") for x in [c,d]]
        if r['numerical_failure'] or s['numerical_failure']:unknownpairs.append(key[:-1])
        same=all(r[k]==s[k] for k in ['success','collision','timeout','deadlock','numerical_failure','episode_length'])
        with np.load(a['trace_path']) as za,np.load(b['trace_path']) as zb:
            same=bool(same and all(np.array_equal(za[k],zb[k]) for k in ['positions','velocities','actions','goals']))
        if not same:mismatches.append(key[:-1])
    out['determinism']=dict(pairs=640,exact=not mismatches,mismatches=mismatches,unknown_pairs=unknownpairs);write(HERE/'A_RESULTS.json',out)
    lines=['# 阶段 A','',f"两 seed 物理轨迹逐元素一致：{not mismatches}；unknown 对数 {len(unknownpairs)}。",'', '| 管线 | Controller | safety | F10 | F13 | S300 | S500 | S500−学习型参考 [95%CI] |','|---|---|---:|---:|---:|---:|---:|---|']
    for pipe in ['local','central']:
        for n in ['Z','G']:
            e=out[pipe]['controls'][n]['arms'];d=e['S500']['minus_reference'];lines.append('| '+pipe+' | '+n+' | '+' | '.join(f"{e[a]['mean']:.2%}" for a in ARMS)+f" | {d['mean']:.2%} {d['CI_unknown_envelope']} |")
        lines+=['',f"{pipe}：参考 S500={out[pipe]['reference_S500']['mean']:.2%}；{out[pipe]['controls']['Z']['S500_interpretation']}。",'']
    (HERE/'A_RESULTS.md').write_text('\n'.join(lines)+'\n');source_inventory('imports_analysis_A.json')

def phaseB():
    y,u=collect('fair',NAMES,['RHR-fair']);sy,su=historical(False);comparison=paired(sy,su,y,u)[0];gy,gu=collect('fair',['G'],['RHR-fair']);gbase,gubase=collect('A',['G'],['S500'],'local');gcompare=paired(gbase,gubase,gy,gu)[0];out=dict(selected=read(HERE/'RHR_SELECTED.json'),students_RHR=paired(y,u)[0],G_RHR=paired(gy,gu)[0],students_S500_minus_RHR=comparison['minus_reference'],G_S500_minus_RHR=gcompare['minus_reference']);out['students_interpretation']='S500 优于公平调参 RHR' if out['students_S500_minus_RHR']['CI_unknown_envelope'][0]>0 else '尚不能证明 S500 优于公平调参 RHR；描述性比较，未设等效界值';write(HERE/'B_RESULTS.json',out);(HERE/'B_RESULTS.md').write_text('# 阶段 B\n\n'+json.dumps(clean(out),indent=2,ensure_ascii=False)+'\n');source_inventory('imports_analysis_B.json')

def possible(lo,hi):
    # Exact tie-breaking matters: a later arm cannot win an exact tie with an earlier arm.
    n=lo.shape[-1];a=[]
    for k in range(n):
        left=np.max(lo[...,:k],axis=-1) if k else np.full(lo.shape[:-1],-np.inf);right=np.max(lo[...,k+1:],axis=-1) if k+1<n else np.full(lo.shape[:-1],-np.inf);a.append((hi[...,k]>left)&(hi[...,k]>=right))
    return np.stack(a,-1)
def take(a,k):return np.take_along_axis(a,np.asarray(k)[...,None],axis=-1)[...,0]
def choose_evaluate(train,trainu,test,testu):
    k=np.argmax(train,-1);p=possible(train,train+trainu);return take(test,k),np.min(np.where(p,test,np.inf),-1),np.max(np.where(p,test+testu,-np.inf),-1)
def headroom(y,u):
    # C,S,seed,18; controller crossfit in fixed state folds, state crossfit in seed folds.
    baseline=y[...,-1].mean(2);baseu=u[...,-1].mean(2);C=len(y);pc=np.zeros((C,32));pcl=pc.copy();pch=pc.copy();ps=pc.copy();psl=pc.copy();psh=pc.copy()
    for fold in [0,1]:
        train=np.arange(32)%2!=fold;test=~train;a,b,c=choose_evaluate(y[:,train].mean((1,2))[:,None,:],u[:,train].mean((1,2))[:,None,:],y[:,test].mean(2),u[:,test].mean(2));pc[:,test]=a;pcl[:,test]=b;pch[:,test]=c
    for seed in [0,1]:
        a,b,c=choose_evaluate(y[:,:,seed],u[:,:,seed],y[:,:,1-seed],u[:,:,1-seed]);ps+=a/2;psl+=b/2;psh+=c/2
    sample_controller=np.max(y.mean((1,2)),-1)-baseline.mean(1);sample_state=np.max(y.mean(2),-1)-baseline
    out={};point=dict(controller=pc-baseline,state=ps-baseline);low=dict(controller=pcl-baseline-baseu,state=psl-baseline-baseu);high=dict(controller=pch-baseline,state=psh-baseline)
    for mode in ['controller','state']:
        out[mode]=dict(mean=float(point[mode].mean()),identification_interval=[float(low[mode].mean()),float(high[mode].mean())])
    rng=np.random.default_rng(2026103032);ic=rng.integers(0,C,(5000,C));ds=rng.integers(0,32,(5000,C,32));bstate=point['state'][ic[:,:,None],ds].mean((1,2));blstate=low['state'][ic[:,:,None],ds].mean((1,2));bhstate=high['state'][ic[:,:,None],ds].mean((1,2))
    bc=[];blc=[];bhc=[]
    even=np.arange(0,32,2);odd=np.arange(1,32,2)
    for b in range(5000):
        ids=ic[b];draws=[rng.choice(even,(C,16)),rng.choice(odd,(C,16))];vals=[];lows=[];highs=[]
        for f in [0,1]:
            tr=draws[1-f];te=draws[f];train=y[ids[:,None],tr].mean((1,2));trainu=u[ids[:,None],tr].mean((1,2));test=y[ids[:,None],te].mean(2);testu=u[ids[:,None],te].mean(2);a,l,h=choose_evaluate(train[:,None,:],trainu[:,None,:],test,testu);base=test[...,-1];bu=testu[...,-1];vals.append((a-base).mean());lows.append((l-base-bu).mean());highs.append((h-base).mean())
        bc.append(np.mean(vals));blc.append(np.mean(lows));bhc.append(np.mean(highs))
    for mode,b,l,h in [('controller',bc,blc,bhc),('state',bstate,blstate,bhstate)]:out[mode].update(CI95=np.quantile(b,[.025,.975]).tolist(),CI_unknown_envelope=[float(np.quantile(l,.025)),float(np.quantile(h,.975))])
    out['in_sample_upper']=dict(controller_mean=float(sample_controller.mean()),state_mean=float(sample_state.mean()),controller_unknown_upper=float((np.max((y+u).mean((1,2)),-1)-baseline.mean(1)).mean()),state_unknown_upper=float((np.max((y+u).mean(2),-1)-baseline).mean()))
    out['baseline_S500']=float(baseline.mean());out['unknown']=int(u.sum());return out

def phaseC():
    y,u=collect('C',NAMES+['Z'],MENU,'local');student=headroom(y[:14],u[:14]);zero=headroom(y[14:],u[14:]);decision=any(student[m]['mean']>=.02 and student[m]['CI_unknown_envelope'][0]>0 for m in ['controller','state']);out=dict(students=student,Z=zero,next_decentralized_critic=decision,decision='下一轮统一测量 D-local controller 表示并训练分散式 critic；本轮不训练' if decision else '在所测 Ring 分布与 K16 菜单中，固定 S500 已接近最优，生成器与 critic 无已证实的发挥空间',candidate_rates={n:{a:float(y[i,...,j].mean()) for j,a in enumerate(MENU)} for i,n in enumerate(NAMES+['Z'])});write(HERE/'C_RESULTS.json',out);np.savez_compressed(HERE/'C_ARRAYS.npz',success_lower=y,unknown=u,names=NAMES+['Z'],arms=MENU);(HERE/'C_RESULTS.md').write_text('# 阶段 C\n\n'+json.dumps(clean(out),indent=2,ensure_ascii=False)+'\n');source_inventory('imports_analysis_C.json')
if __name__=='__main__':{'A':phaseA,'B':phaseB,'C':phaseC}[sys.argv[1]]()
