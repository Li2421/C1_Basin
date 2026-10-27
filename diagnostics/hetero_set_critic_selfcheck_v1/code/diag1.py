from tab import *
from scipy.stats import spearmanr
import critic as CR
sp=json.load(open('SPLIT.json'));VA=sp['val'];G=json.load(open('GEN_FROZEN.json'));Z=np.load('diag_logits.npz')
T=load(['design.jsonl','extra.jsonl','fresh.jsonl']);F=list(range(8,24))
Y={v:np.full((3,24,16),np.nan) for v in VA}
for l in open('gen.jsonl'):d=json.loads(l);Y[d['scene']][d['controller'],d['state'],d['proposal']]=d['success']
def soft(f):p=np.exp(f-f.max(-1,keepdims=True));return (p/p.sum(-1,keepdims=True)).mean((0,1))
rows=[]
print(f"{'scene':26s} {'rho_prop':>8s} {'rho_menu':>8s} | pick_soft  pick_logit  best | union_soft union_logit")
tot={k:[] for k in ['soft','logit','best','u_soft','u_logit','menu_soft']}
for v in VA:
 fp=Z[v+'__prop'].mean(0);fm=Z[v+'__menu'].mean(0)   # (3,24,K)
 yp=Y[v][:,F].mean((0,1));ym=T[v][:,F].mean((0,1))
 rp=spearmanr(fp.mean((0,1)),yp).correlation;rm=spearmanr(fm.mean((0,1)),ym).correlation
 s1=int(soft(fp).argmax());s2=int(fp.mean((0,1)).argmax())
 fu=np.concatenate([fp,fm],-1);yu=np.concatenate([yp,ym]);u1=int(soft(fu).argmax());u2=int(fu.mean((0,1)).argmax());ms=int(soft(fm).argmax())
 tot['soft'].append(yp[s1]);tot['logit'].append(yp[s2]);tot['best'].append(yp.max());tot['u_soft'].append(yu[u1]);tot['u_logit'].append(yu[u2]);tot['menu_soft'].append(ym[ms])
 print(f"{v:26s} {rp:8.2f} {rm:8.2f} | {yp[s1]*100:6.1f} {yp[s2]*100:9.1f} {yp.max()*100:7.1f} | {yu[u1]*100:6.1f}{' (menu)' if u1>=16 else ' (gen) '} {yu[u2]*100:6.1f}{' (menu)' if u2>=16 else ' (gen) '}")
print({k:round(np.mean(x)*100,2) for k,x in tot.items()})
# duplicate splitting: pairwise distance of proposals and of top ones
for v in VA:
 P=np.array(G[v]['proposals']);Pn=(P-CR.CENTER)/CR.RADIUS;D=np.linalg.norm(Pn[:,None]-Pn[None],axis=-1);yp=Y[v][:,F].mean((0,1))
 b=int(yp.argmax());print(v,'best prop',np.round(P[b],2).tolist(),'#props within .15 of best',int((D[b]<.15).sum()),'| side comps',np.round(P[:,1],2).tolist())
