"""Cross-controller oracle on exact within-state physical actions; donor refit per draw."""
from common import *
from menu_c6 import METHODS
import csv

def key(a):
    def bits(x):return tuple(float(v).hex() for v in x)
    early=bits(a['eta_a']);late=bits(a['eta_b']);rq=a.get('late_source','given')=='requery';trigger=a.get('trigger','none');mode=a['switch_mode']
    if not rq and early==late:
        return ('constant',early) if trigger=='none' else ('trigger_constant',early,trigger)
    switch=float(a.get('progress_threshold',.9681620884830543)).hex() if mode=='progress' else int(a['switch_step'])
    return ('requery' if rq else 'given',early,() if rq else late,mode,switch,trigger)

def definition():
    write(HERE/'TYPE_INFORMATION_ORACLE_DEFINITION.json',dict(utc=utc(),identity='Same physical state and full float64 eta bits, schedule mode, exact threshold/fixed step, trigger and requery semantics; constants canonicalized; no cross-state proposal-index alignment',candidates='Only physical actions actually evaluated for the held-out controller at that state; both donor pools must support an action',same_type='Other actual controller identities of the same type, excluding every bootstrap copy of held-out controller',other_types='Equal average over supported other types, equal donor-controller contribution within a type',selection='Largest donor success lower, ties canonical physical key; no common-supported action => both choose Safety with explicit unscored flag',unknown='Donor choices use unknown0; held-out paired lower/upper bounds conditional on these choices, identical selected physical actions cancel',bootstrap='5000 paired target type->controller->state draws, q0/1 clustered; source catalogue remains the known19 types. Donor controller/state bootstrap refits choices on every draw, coupled to target resampling inside the target group; outside-group donor controllers resampled separately. Every held-out actual controller excluded even when repeatedly sampled.',support='Sparse evaluated-action common-support empirical type-information reference, not a theoretical upper bound for every possible action',no_model_predictions_as_labels=True))
    seal('TYPE_INFORMATION_ORACLE_DEFINITION.receipt.json',[HERE/'TYPE_INFORMATION_ORACLE_DEFINITION.json',HERE/'oracle_c6.py'],before_fresh_evaluation_outcomes=True)

def load():
    with np.load(HERE/'NEW_EVAL_BUNDLE.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    nn=list(d['names']);rows=read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states'];lookup={(r['controller'],r['state_index']):r['methods'] for r in rows};pack=[]
    for si in range(24):
        keys=[[key(lookup[(n,si)][m]) for m in METHODS] for n in nn];unique=sorted(set(k for kk in keys for k in kk),key=repr);index={k:i for i,k in enumerate(unique)};K=len(unique);Y=np.zeros((122,K));U=np.zeros_like(Y);M=np.zeros_like(Y);ids=np.full((122,8),-1,np.int32);safety=np.zeros(122,np.int32)
        for ci in range(122):
            observed=sorted({index[k] for k in keys[ci]});ids[ci,:len(observed)]=observed;safety[ci]=observed.index(index[keys[ci][7]])
            for mi,k in enumerate(keys[ci]):
                ai=index[k];lo=float(np.mean(d['labels'][ci,si,mi]==1));up=float(np.mean(d['labels'][ci,si,mi]!=0))
                if M[ci,ai]:assert Y[ci,ai]==lo and U[ci,ai]==up,'Inconsistent equivalent-action labels'
                Y[ci,ai]=lo;U[ci,ai]=up;M[ci,ai]=1
        pack.append(dict(Y=Y,U=U,M=M,ids=ids,safety=safety,keys=unique))
    return d,pack

def weights(types,mask,B,rng,point=False):
    N=len(types);S=24;tt=sorted(set(types[mask]));target=np.zeros((B,N,S));donor=np.zeros_like(target)
    if point:
        for t in tt:target[:,mask&(types==t)]=1/(len(tt)*np.sum(mask&(types==t))*S)
        for t in sorted(set(types)):donor[:,types==t]=1/(np.sum(types==t)*S)
        return target,donor
    draw=rng.choice(tt,(B,len(tt)))
    def add(out,reps,pool,denominator):
        if not len(pool) or not len(reps):return
        cc=rng.choice(pool,(len(reps),len(pool)));ss=rng.integers(S,size=(len(reps),len(pool),S));rr=np.broadcast_to(np.asarray(reps)[:,None,None],ss.shape);ci=np.broadcast_to(cc[:,:,None],ss.shape);np.add.at(out,(rr,ci,ss),1/(denominator*S))
    for t in tt:
        br,_=np.where(draw==t);pool=np.flatnonzero(mask&(types==t));add(target,br,pool,len(tt)*len(pool))
    for t in sorted(set(types)):
        inside=np.flatnonzero(mask&(types==t));outside=np.flatnonzero(~mask&(types==t));nc=len(inside)+len(outside)
        if len(inside):
            total=target[:,inside].sum((1,2));valid=total>0
            donor[:,inside]=np.where(valid[:,None,None],target[:,inside]/np.maximum(total[:,None,None],1e-30)*(len(inside)/nc),0)
            add(donor,np.flatnonzero(~valid),inside,nc)
        add(donor,np.arange(B),outside,nc)
    assert np.allclose(target.sum((1,2)),1.)
    return target,donor

def evaluate(pack,types,target,donor):
    tt=sorted(set(types));ti=np.asarray([tt.index(t) for t in types]);B=len(target);output=np.zeros((B,4));ci=np.arange(122)
    for si,p in enumerate(pack):
        Y,U,M,ids=p['Y'],p['U'],p['M'],p['ids'];ii=np.maximum(ids,0);K=Y.shape[1];num=np.empty((B,len(tt),K));den=np.empty_like(num)
        for j,t in enumerate(tt):
            mask=types==t;ww=donor[:,mask,si];num[:,j]=ww@Y[mask];den[:,j]=ww@M[mask]
        pr=np.divide(num,den,out=np.zeros_like(num),where=den>1e-20);support=den>1e-20;tp=pr.sum(1);tc=support.sum(1)
        nd=num[:,ti[:,None],ii];dd=den[:,ti[:,None],ii];own=donor[:,:,si,None];heldY=Y[ci[:,None],ii];heldU=U[ci[:,None],ii]
        sameDen=dd-own;ps=np.divide(nd-own*heldY[None],sameDen,out=np.zeros_like(nd),where=sameDen>1e-20);otherDen=tc[:,ii]-support[:,ti[:,None],ii];po=np.divide(tp[:,ii]-pr[:,ti[:,None],ii],otherDen,out=np.zeros_like(nd),where=otherDen>0)
        common=(sameDen>1e-20)&(otherDen>0)&(ids[None]>=0);missing=~common.any(-1);sa=np.argmax(np.where(common,ps,-np.inf),-1);oa=np.argmax(np.where(common,po,-np.inf),-1);sa=np.where(missing,p['safety'][None],sa);oa=np.where(missing,p['safety'][None],oa)
        sy=heldY[ci[None],sa];oy=heldY[ci[None],oa];su=heldU[ci[None],sa];ou=heldU[ci[None],oa];equal=np.take_along_axis(np.broadcast_to(ids,(B,*ids.shape)),sa[:,:,None],-1)[...,0]==np.take_along_axis(np.broadcast_to(ids,(B,*ids.shape)),oa[:,:,None],-1)[...,0];w=target[:,:,si]
        output[:,0]+=np.sum(w*(sy-oy),1);output[:,1]+=np.sum(w*np.where(equal,0,sy-ou),1);output[:,2]+=np.sum(w*np.where(equal,0,su-oy),1);output[:,3]+=np.sum(w*missing,1)
    return output

def run(worker,workers):
    d,pack=load();types=d['types'];groups=read(HERE/'GROUP_MEMBERSHIP.json')['groups'];names=d['names']
    for gi,(group,members) in enumerate(groups.items()):
        if gi%workers!=worker:continue
        path=HERE/'oracle_groups'/f'{gi:02}.json'
        if path.exists():continue
        mask=np.isin(names,members)
        if not mask.any():write(path,dict(group=group,gap='empty subgroup'));continue
        rng=np.random.default_rng(2026102064+10000+gi);target,donor=weights(types,mask,1,rng,True);values=[evaluate(pack,types,target,donor)]
        for begin in range(0,5000,25):
            target,donor=weights(types,mask,min(25,5000-begin),rng);values.append(evaluate(pack,types,target,donor))
            if begin%500==0:print('Oracle donor-refit bootstrap',group,begin,flush=True)
        values=np.concatenate(values);npz=path.with_suffix('.npz');npz.parent.mkdir(exist_ok=True);np.savez_compressed(npz,bootstrap=values)
        write(path,dict(utc=utc(),group=group,point=float(values[0,0]*100),CI95=np.quantile(values[1:,0]*100,[.025,.975]),unknown_bounds=(values[0,1:3]*100),unknown_lower_CI95=np.quantile(values[1:,1]*100,[.025,.975]),unknown_upper_CI95=np.quantile(values[1:,2]*100,[.025,.975]),unscored_fallback_fraction=float(values[0,3]),bootstrap_fallback_mean=float(values[1:,3].mean()),draws=5000,donor_choices_refitted_each_draw=True,source_catalogue_fixed_known19=True,held_out_controller_always_excluded=True,sparse_common_support=True,bootstrap_path=str(npz)))
    write(HERE/f'ORACLE_WORKER_{worker}_COMPLETE.json',dict(utc=utc(),done=True))

def collect():
    groups=read(HERE/'GROUP_MEMBERSHIP.json')['groups'];results={};paths=[]
    for i,g in enumerate(groups):
        p=HERE/'oracle_groups'/f'{i:02}.json';r=read(p);assert r['group']==g;results[g]=r;paths.append(p)
        if 'bootstrap_path' in r:paths.append(Path(r['bootstrap_path']))
    write(HERE/'TYPE_INFORMATION_ORACLE_RESULTS.json',dict(utc=utc(),groups=results,definition=read(HERE/'TYPE_INFORMATION_ORACLE_DEFINITION.json')));r=read(HERE/'T1_T7_RESULTS.json');table={}
    for g,row in r['groups'].items():
        cc=row.get('comparisons',{});table[g]=dict(layers={METHODS[0]:cc.get('T1_vs_Safety'),METHODS[1]:cc.get('T2_online'),METHODS[2]:cc.get('T3_offline')},perfect_type_information_empirical_reference=results[g])
    write(HERE/'INFORMATION_VALUE_NUMERIC.json',dict(utc=utc(),groups=table,layer_information={METHODS[0]:'state only',METHODS[1]:'current episode progress, used online to trigger correction',METHODS[2]:'700-step offline outcome-free probe features, used to score eta/threshold/trigger actions'}));seal('TYPE_INFORMATION_ORACLE.receipt.json',[HERE/'TYPE_INFORMATION_ORACLE_RESULTS.json',HERE/'INFORMATION_VALUE_NUMERIC.json',HERE/'TYPE_INFORMATION_ORACLE_DEFINITION.receipt.json',*paths],numeric_only=True)
    seal('NUMERIC_RESULTS.receipt.json',[HERE/n for n in ('T1_T7_STATISTICS.receipt.json','TYPE_INFORMATION_ORACLE.receipt.json','T1_T7_RESULTS.json','TYPE_INFORMATION_ORACLE_RESULTS.json','INFORMATION_VALUE_NUMERIC.json','AXIS_TABLE.csv','CONTROLLER_RESULTS.csv','EVENT_AND_CHOICES.json')],before_any_plot=True)

if __name__=='__main__':run(int(sys.argv[2]),int(sys.argv[3])) if sys.argv[1]=='run' else globals()[sys.argv[1]]()
