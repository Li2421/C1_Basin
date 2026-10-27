"""Preregistered finite candidate screening, subset closure and frozen Oracle@16."""
from dispatch import *
from shared_control.basis_dictionary import ALL,NEW,SIGNED
from scipy.stats import qmc
import itertools

NAMES={'G':'朝目标（限速目标残差）','P':'横向放大（目标方向正交的流残差）','R':'互斥分离（接触门控的平均分离）','T1':'目标右侧旋转','T2':'群中心旋转','T3':'冲突对旋转','T4':'墙切向','T5':'活跃约束切向投影','P2':'原始控制输出的横向残差','R2':'按松弛加权的分离','R3':'最紧一对的分离','Y1':'让行者后退','Y2':'让行者侧让','Y3':'让行者离开通行者路线','Y4':'让行者原地等待','S1':'让行者后退、通行者前进','C1':'通道轴向','C2':'局部净空梯度'}

def support(sp):return tuple(k for k,v in zip(ALL[:3],sp['eta'])if v!=0.)+tuple(k for k in NEW if sp.get('coefficients',{}).get(k,0.)!=0.)
def mask(sp):return sum(1<<ALL.index(k)for k in support(sp))
def names(m):return [k for j,k in enumerate(ALL)if m>>j&1]
def key(r):return (r['type_id'],r['controller'],r['state_index'])

def sobol(cols,n,seed):
    cols=[k for k in ALL if k in cols];d=len(cols)
    if not d:return [dict(eta=[0.,0.,0.],coefficients={})]
    samples=qmc.Sobol(d,scramble=True,seed=seed).random_base2(int(math.ceil(math.log2(n))))[:n]
    specs=[]
    for sample in samples:
        values={}
        for name,u in zip(cols,sample):
            lo,hi=(.5,1.25)if name=='G'else(-.5,.5)if name=='P'else(0.,.75)if name=='R'else(-1.,1.)if name in SIGNED else(0.,1.)
            values[name]=float(lo+(hi-lo)*u)
        specs.append(dict(eta=[values.get(k,0.)for k in ALL[:3]],coefficients={k:values[k]for k in NEW if k in values}))
    return specs

def cohorts(frozen,indices):
    out=[]
    for e in frozen['entries']:
        if e['status']!='applicable':continue
        for ctl in controllers():
            if ctl['type_code']not in e['applicable_controllers']:continue
            for i in indices:out.append((e['scene'],ctl,i))
    return out

def specs_s1():
    prior=read(ROOT/'diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/ACTIONS.json')['lit_swap8_free']
    result=[dict(eta=prior[f'eta{i}']['eta_a'],coefficients={})for i in range(16)]
    for name in NEW:
        for gain in ((-.75,.75)if name in SIGNED else(.5,1.)):
            result.append(dict(eta=[.875,0.,.375],coefficients={name:gain}))
    assert len(result)==46;return result

def successful_supports(rows,cases):
    keys={key(c):j for j,c in enumerate(cases)};coverage={}
    for r in rows:
        if r['success']and key(r)in keys:
            m=mask(r['spec']);coverage[m]=coverage.get(m,0)|(1<<keys[key(r)])
    dp=[0]*(1<<len(ALL))
    for m,bits in coverage.items():dp[m]|=bits
    # Union all successful candidate supports contained in each candidate family.
    for j in range(len(ALL)):
        bit=1<<j
        for m in range(1<<len(ALL)):
            if m&bit:dp[m]|=dp[m^bit]
    return dp

def summarize_supports(rows,cohort):
    cases=[dict(type_id=s['type_id'],controller=ctl['type_code'],state_index=i)for s,ctl,i in cohort]
    dp=successful_supports(rows,cases);maximum=dp[-1];covers=[m for m,b in enumerate(dp)if b==maximum]
    dim=min(m.bit_count()for m in covers);small=[m for m in covers if m.bit_count()==dim]
    bytype={}
    for t in sorted({c['type_id']for c in cases}):
        bits=sum(1<<j for j,c in enumerate(cases)if c['type_id']==t)&maximum
        ms=[m for m,b in enumerate(dp)if b&bits==bits];d=min(m.bit_count()for m in ms)
        mini=[m for m in ms if m.bit_count()==d];required=(1<<len(ALL))-1
        for m in mini:required&=m
        bytype[str(t)]=dict(covered_cases=bits.bit_count(),dimension=d,minimal_supports=[names(m)for m in mini],required_columns=names(required),not_possible_proof=False)
    result=dict(observed_covered_cases=maximum.bit_count(),search_cases=len(cases),global_dimension=dim,global_minimal_supports=[names(m)for m in small],per_type=bytype,meaning='Subset closure of finite successful witnesses; not a continuous-space impossibility/minimality theorem.')
    return result,dp

def all_rows(stages):
    rows=[]
    for stage in stages:
        p=HERE/f'{stage}_planned.json'
        if not p.exists():continue
        for c in read(p)['exact_configs']:
            q=HERE/'raw'/f"{c['rollout_uid']}.json"
            if q.exists():rows.append(read(q))
    return list({r['rollout_uid']:r for r in rows}.values())

def rank_families(rows,dp,maxdim=5):
    collision={};times={}
    for r in rows:
        m=mask(r['spec']);collision[m]=collision.get(m,0)+r['collision'];times.setdefault(m,[]).append(r['truncated_time'])
    def merit(m):return (-dp[m].bit_count(),collision.get(m,0),sum(times.get(m,[float('inf')]))/len(times.get(m,[0])),m)
    return {d:min((m for m in range(1<<len(ALL))if m.bit_count()==d),key=merit)for d in range(1,maxdim+1)}

def search(frozen):
    cohort=cohorts(frozen,range(4));stages=[];basecases=[case(s,c,i,purpose='search')for s,c,i in cohort]
    stage='S1';stages.append(stage)
    execute([case(s,c,i,sp,purpose='search')for s,c,i in cohort for sp in specs_s1()],stage)
    pilot=[]
    for e in frozen['entries']:
        if e['status']!='applicable':continue
        for ctl in controllers():
            if ctl['type_code']not in e['applicable_controllers']:continue
            rr=[]
            for h in read(HERE/'CALIBRATION.json')['history'][str(e['type_id'])]:
                if h['scene']['definition_sha256']!=e['scene']['definition_sha256']:continue
                for uid in h['metrics']['raw_rollout_uids']:
                    r=read(HERE/'raw'/f'{uid}.json')
                    if r['purpose']=='calibration_safety'and r['controller']==ctl['type_code']:rr.append(r)
            if not rr:continue
            worst=max(rr,key=lambda r:(sum(r['final_route_distances'])if r.get('final_route_distances')is not None else float('inf'),-r['state_index']))
            pilot.append((e['scene'],ctl,worst['state_index']))
    pair_specs=[]
    for j,pair in enumerate(itertools.combinations(NEW,2)):
        sp=sobol(pair,1,2026102802+j)[0];sp['eta']=[.875,0.,.375];pair_specs.append(sp)
    stage='S2';stages.append(stage)
    execute([case(s,c,i,sp,purpose='search')for s,c,i in pilot for sp in pair_specs],stage)
    rows=all_rows(stages);covered={key(r)for r in rows if r['success']};uncovered=[x for x in cohort if(x[0]['type_id'],x[1]['type_code'],x[2])not in covered]
    rank=[]
    for sp in pair_specs:
        rr=[r for r in rows if digest(r['spec'])==digest(sp)];rank.append(( -len({r['type_id']for r in rr if r['success']}),-sum(r['success']for r in rr),sum(r['collision']for r in rr),digest(sp),sp))
    top=[tuple(x[4]['coefficients'])for x in sorted(rank,key=lambda x:x[:4])[:12]]
    rng=np.random.default_rng(2026102802);sets=[tuple(('G','R')+p)for p in top]
    for j in range(16):sets.append(tuple(k for k in ALL if k in set(rng.choice(ALL,size=int(rng.integers(3,6)),replace=False))))
    sets.extend([ALL,ALL]);plans=[]
    for j,cols in enumerate(sets):
        for sp in sobol(cols,4,2026103000+j):
            for s,c,i in uncovered:plans.append(case(s,c,i,sp,purpose='search'))
    stage='S3';stages.append(stage);execute(plans[:8000],stage)
    rows=all_rows(stages);minimum,dp=summarize_supports(rows,cohort);dump(HERE/'MINIMAL_SUPPORTS_before_S4.json',minimum)
    ranked=rank_families(rows,dp);pool=[]
    # The prompt asks for 1–5 dimension results. If no successful low-dimensional
    # support has yet appeared, the same finite S4 budget probes it explicitly.
    for d in range(1,min(6,max(2,minimum['global_dimension']))):pool.append(ranked[d])
    previous=read(ROOT/'diagnostics/paper_baselines_20261028/basis/FINAL_FROZEN.json')['best']['spec']
    if mask(previous) not in pool and mask(previous).bit_count()<max(2,minimum['global_dimension']):pool.append(mask(previous))
    for m in sorted(range(1<<len(ALL)),key=lambda m:(-dp[m].bit_count(),m.bit_count(),m)):
        if 0<m.bit_count()<max(2,minimum['global_dimension'])and m not in pool:pool.append(m)
        if len(pool)>=6:break
    queues=[]
    for j,m in enumerate(pool[:6]):
        sub=[x for n,x in enumerate(cohort)if not dp[m]>>n&1]
        queues.append([case(s,c,i,sp,purpose='search')for sp in sobol(names(m),16,2026104000+j)for s,c,i in sub])
    # Round robin the predeclared support queues so one family cannot consume
    # the complete 2,000 episode budget before smaller supports are examined.
    plans=[]
    for batch in itertools.zip_longest(*queues):plans.extend(c for c in batch if c is not None)
    stage='S4';stages.append(stage);execute(plans[:2000],stage)
    rows=all_rows(stages);minimum,dp=summarize_supports(rows,cohort);dump(HERE/'MINIMAL_SUPPORTS.json',minimum)
    return cohort,rows,minimum,dp

def freeze_families(cohort,rows,minimum,dp):
    p=HERE/'FAMILIES_FROZEN.json'
    if p.exists():return read(p)
    ranked=rank_families(rows,dp);selected=[7]+[ranked[d]for d in range(1,6)]
    globalmask=sum(1<<ALL.index(k)for k in minimum['global_minimal_supports'][0])
    selected.append(globalmask)
    previous=read(ROOT/'diagnostics/paper_baselines_20261028/basis/FINAL_FROZEN.json')['best']['spec'];selected.append(mask(previous))
    selected=list(dict.fromkeys(selected))
    if len(selected)>7:
        # This is an explicit prompt conflict, not a performance-based deletion.
        dump(HERE/'FREEZE_CONFLICT.json',dict(required_unique_families=[names(m)for m in selected],maximum=7,reason='B3 + dimension1..5 + global minimum + prior best yields eight unique required families.'))
        raise RuntimeError('Prompt freeze conflict: eight mandatory unique families versus maximum seven; no silent deletion.')
    families=[]
    for j,m in enumerate(selected):
        groups={}
        for r in rows:
            if mask(r['spec'])&~m==0:groups.setdefault(digest(r['spec']),[]).append(r)
        order=sorted(groups,key=lambda h:(-sum(r['success']for r in groups[h]),sum(r['collision']for r in groups[h]),sum(r['truncated_time']for r in groups[h])/len(groups[h]),h))
        specs=[groups[h][0]['spec']for h in order[:8]]
        # Sparse/deletion families may have fewer than eight unique observed
        # settings. Their missing slots are explicitly untested Sobol settings.
        if len(specs)<8:specs+=sobol(names(m),8-len(specs),2026105000+j)
        specs+=sobol(names(m),8,2026105100+j)
        if m==mask(previous)and digest(previous)not in {digest(s)for s in specs}:specs[7]=previous
        families.append(dict(columns=names(m),column_names=[NAMES[k]for k in names(m)],dimension=m.bit_count(),candidate_pool=specs,fixed=specs[0],search_covered=dp[m].bit_count(),pool_order='Best up to8 observed settings then Sobol fill; +8 new Sobol. Fixed=best observed search setting; untested fills are never credited with search success.'))
    frozen=dict(families=families,selection_by_dimension={str(d):names(m)for d,m in ranked.items()},global_minimum=names(globalmask),prior_best=previous,search_cases=len(cohort),oracle_K=16,seed=0,holdout_states=[4,5,6,7],meaning='Finite candidate pools; no mathematical optimality claim.')
    dump(p,frozen);return frozen

def holdout(frozen,families):
    cohort=cohorts(frozen,range(4,8));cs=[];labels={}
    for j,f in enumerate(families['families']):
        for s,c,i in cohort:
            for k,sp in enumerate(f['candidate_pool']):
                cc=case(s,c,i,sp,purpose='holdout');cs.append(cc);labels.setdefault(str(j),[]).append(dict(uid=cc['rollout_uid'],candidate=k,type_id=s['type_id'],controller=c['type_code'],state_index=i))
    for name,sp in [('safety',dict(eta=[0.,0.,0.],coefficients={})),('right_goal_reference',dict(eta=[0.,0.,0.],coefficients={},reference='right_goal')),('centre_rotation_reference',dict(eta=[0.,0.,0.],coefficients={},reference='centre_rotation'))]:
        for s,c,i in cohort:
            cc=case(s,c,i,sp,purpose='holdout_reference');cs.append(cc);labels.setdefault(name,[]).append(dict(uid=cc['rollout_uid'],candidate=0,type_id=s['type_id'],controller=c['type_code'],state_index=i))
    dump(HERE/'HOLDOUT_LABELS.json',labels);return execute(cs,'holdout')

def main():
    frozen=read(HERE/'SCENES_FROZEN.json');cohort,rows,minimum,dp=search(frozen)
    families=freeze_families(cohort,rows,minimum,dp);holdout(frozen,families)

if __name__=='__main__':main()
