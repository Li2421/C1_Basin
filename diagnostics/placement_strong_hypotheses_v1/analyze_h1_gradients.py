"""Check G->S->goal-margin chain and first-order root predictions."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')


def path(j):
    stem=f"{j['uid']}_root{j['root_chain']}_eval{j['eval_chain']}_{j['kind']}_e{j['epsilon']}_a{j['axis']}_s{j['sign']}"
    return HERE/'h1_gradient_results'/stem


def load(j):
    p=path(j);r=json.loads(Path(str(p)+'.json').read_text())
    with np.load(Path(str(p)+'.npz')) as z:a={k:z[k].copy() for k in z.files}
    return r,a


def signature(r,arr,goals):
    kind='goal' if r['goal_ever_margin']<=r['safety_always_margin'] else 'safety'
    step=r['goal_active_step'] if kind=='goal' else r['safety_active_step']
    if kind=='goal':
        pos=arr['positions'][step];agent=int(np.argmax(np.linalg.norm(pos-goals,axis=1)))
    else:agent=None
    return kind,step,agent


def main():
    p=json.loads((HERE/'h1_gradient_protocol.json').read_text())
    states={s['uid']:s for s in json.loads((HARD/'confirmation_manifest.json').read_text())['states']}
    rows=[]
    for c in p['centers']:
        uid=c['uid'];goals=np.asarray(states[uid]['physical']['goals'])
        for chain in ('TT','FF'):
            jobs=[x for x in p['jobs'] if x['uid']==uid and x['root_chain']==c['root_chain'] and x['eval_chain']==chain]
            j0=next(x for x in jobs if x['kind']=='center');base,arr=load(j0)
            sig0=signature(base,arr,goals)
            estimates={}
            for eps in (.0005,.002):
                grad=[];G=[];S=[];sigs=[];goal_chain=[]
                for axis in range(3):
                    m=next(x for x in jobs if x['epsilon']==eps and x['axis']==axis and x['sign']==-1)
                    q=next(x for x in jobs if x['epsilon']==eps and x['axis']==axis and x['sign']==1)
                    rm,am=load(m);rp,ap=load(q)
                    sigs += [signature(rm,am,goals),signature(rp,ap,goals)]
                    grad.append((rp['rho']-rm['rho'])/(2*eps))
                    G.append((ap['actions'][0].ravel()-am['actions'][0].ravel())/(2*eps))
                    t=sig0[1]
                    if len(am['positions'])<=t or len(ap['positions'])<=t:
                        S.append(None);goal_chain.append(None)
                    else:
                        s=(ap['positions'][t]-am['positions'][t])/(2*eps)
                        S.append(s)
                        if sig0[0]=='goal':
                            i=sig0[2];z=arr['positions'][t,i]-goals[i]
                            goal_chain.append(float(-z@ s[i]/np.linalg.norm(z)))
                        else:goal_chain.append(None)
                estimates[eps]=dict(rho_gradient=grad,G=np.stack(G,axis=1).tolist(),
                    S_active=[x.tolist() if x is not None else None for x in S],
                    goal_chain_gradient=goal_chain,
                    all_signatures_same=all(x==sig0 for x in sigs),signatures=sigs)
            g=np.asarray(estimates[.0005]['rho_gradient']);g2=np.asarray(estimates[.002]['rho_gradient'])
            d=np.asarray(c['line_direction']);slope=float(g@d)
            row=dict(uid=uid,root_chain=c['root_chain'],eval_chain=chain,fraction=c['fraction'],
                     center_rho=base['rho'],center_signature=sig0,gradient_stability=float(np.linalg.norm(g-g2)/max(np.linalg.norm(g),1e-12)),
                     slope_along_line=slope,estimates=estimates)
            if all(x is not None for x in estimates[.0005]['goal_chain_gradient']):
                cg=np.asarray(estimates[.0005]['goal_chain_gradient'])
                row['goal_chain_gradient_error']=float(np.linalg.norm(cg-g)/max(np.linalg.norm(g),1e-12))
            rows.append(row)
    roots=json.loads((HERE/'h1_roots.json').read_text())
    pairs=[]
    for uid in sorted({x['uid'] for x in p['centers']}):
        t_tt=next(x['first_root_fraction'] for x in roots if x['uid']==uid and x['chain']=='TT')
        t_ff=next(x['first_root_fraction'] for x in roots if x['uid']==uid and x['chain']=='FF')
        c=next(x for x in p['centers'] if x['uid']==uid and x['root_chain']=='TT');d=np.asarray(c['line_direction'])
        for source,target,t0,t1 in (('TT','FF',t_tt,t_ff),('FF','TT',t_ff,t_tt)):
            row=next(x for x in rows if x['uid']==uid and x['root_chain']==source and x['eval_chain']==target)
            denom=row['slope_along_line']
            pred=t0-row['center_rho']/denom if abs(denom)>1e-12 else None
            true=t1-t0
            pairs.append(dict(uid=uid,source_root=source,target=target,true_signed_fraction_displacement=true,
                true_eta_distance=abs(true)*np.linalg.norm(d),predicted_signed_fraction_displacement=None if pred is None else pred-t0,
                predicted_eta_distance=None if pred is None else abs(pred-t0)*np.linalg.norm(d),
                absolute_fraction_error=None if pred is None else abs(pred-t1),
                displacement_cosine=None if pred is None else float(np.sign((pred-t0)*true)),
                signed_target_normal_displacement=float(true*denom/max(np.linalg.norm(row['estimates'][.0005]['rho_gradient']),1e-12)),
                gradient_stability=row['gradient_stability'],active_signature=row['center_signature'],
                branch_stable=row['estimates'][.0005]['all_signatures_same']))
    result=dict(rows=rows,boundary_predictions=pairs,
                caution='First-order root prediction uses measured target rho and path-sensitive margin gradient, not G alone.')
    (HERE/'h1_gradient_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(pairs,indent=2))


if __name__=='__main__':main()
