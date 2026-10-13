"""Source gate for the full-vector localization control (no held labels)."""
import json
from pathlib import Path
import numpy as np
from .vector_study import HERE,PARENT,SEEDS,read,write,sha

def center(x,axes):
    for a in axes:x=x-x.mean(a,keepdims=True)
    return x

def ci(x):return np.quantile(x,[.025,.975]).tolist()

def skill(p,t):return float(np.mean(2*p*t-p*p))

def run():
    d=dict(np.load(HERE/'dataset.npz'));raw=dict(np.load(HERE/'motion_raw.npz'))
    arms=('current','complete','eta_only')
    assert all((HERE/'fits/joint'/f'{t}_{a}'/f'seed{s}/complete.json').exists() for t in ('return','motion') for a in arms for s in SEEDS)
    models={(t,a):[np.load(HERE/'fits/joint'/f'{t}_{a}'/f'seed{s}/predictions.npz')['probability'] for s in SEEDS] for t in ('return','motion') for a in arms}
    fits={(t,a):[read(HERE/'fits/joint'/f'{t}_{a}'/f'seed{s}/complete.json') for s in SEEDS] for t in ('return','motion') for a in arms}
    rows=[];boots={};saved={};gate_boot=[];half_gate=[[],[]];source_metrics=[]
    for sc in sorted(set(d['scene'])):
        ix=np.flatnonzero((d['scene']==sc)&(d['split']=='validation'))
        ix=np.array(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
        n=len(ix)//4;na=int(d['entity_agent_mask'][ix[0]].sum())*2
        y=raw['vector_motion'][ix][:,[0,5,10,15],:,:na].reshape(n,4,4,4,na)
        assert np.isfinite(y).all()
        h=np.stack((y[:,:,:,:2].mean(3),y[:,:,:,2:].mean(3)),-1);true=h.mean(-1)
        pp={a:np.mean(models['motion',a],0)[ix][:,[0,5,10,15],:na].reshape(n,4,4,na) for a in arms}
        tag=str(sc);saved[tag+'__states']=d['state_uid'][ix][::4];saved[tag+'__truth_halves']=h
        ib=np.random.default_rng(2026101030+len(rows)).integers(n,size=(20000,n));boots[tag+'__indices']=ib
        for a,p in pp.items():
            saved[tag+'__'+a+'_prediction']=p
            for interaction,axes in (('state_eta',(0,2)),('state_controller_eta',(0,1,2))):
                pc=center(p,axes);tc=center(true,axes);hb=[center(h[...,j],axes) for j in range(2)]
                vals=[];halves=[[],[]]
                for start in range(0,len(ib),500):
                    jj=ib[start:start+500];bbaxes=tuple(x+1 for x in axes)
                    pr=center(p[jj],bbaxes);tt=center(true[jj],bbaxes)
                    red=tuple(range(1,pr.ndim))
                    vals.append(np.mean(2*pr*tt-pr*pr,axis=red))
                    for k in range(2):
                        th=center(h[...,k][jj],bbaxes);halves[k].append(np.mean(2*pr*th-pr*pr,axis=red))
                vals=np.concatenate(vals);hh=[np.concatenate(v) for v in halves]
                key=tag+'__'+a+'__'+interaction;boots[key]=vals
                for k in range(2):boots[key+f'__half{k}']=hh[k]
                rows.append(dict(scene=str(sc),N=n,arm=a,interaction=interaction,
                    skill=skill(pc,tc),CI95=ci(vals),half_skill=[skill(pc,t) for t in hb],
                    half_CI95=[ci(v) for v in hh],
                    corr=float(np.corrcoef(pc.ravel(),tc.ravel())[0,1]) if pc.std()>1e-12 and tc.std()>1e-12 else None,
                    Q_MSE=float(np.mean((p-true)**2))))
                if a=='complete' and interaction=='state_eta':
                    gate_boot.append(vals)
                    for k in range(2):half_gate[k].append(hh[k])
        for target in ('return','motion'):
            for a in arms:
                gs=[g for fit in fits[target,a] for g in fit['selected']['groups'] if g['scene']==sc]
                source_metrics.append(dict(scene=str(sc),target=target,arm=a,N=n,
                    state_eta_MSE=float(np.mean([g['state_eta_MSE'] for g in gs])),
                    Q_MSE=float(np.mean([g['Q_MSE'] for g in gs]))))
    joint_ci=ci(np.mean(gate_boot,0));half_ci=[ci(np.mean(v,0)) for v in half_gate]
    seed_better=[fits['motion','complete'][k]['selected']['state_eta_MSE'] < fits['motion','eta_only'][k]['selected']['state_eta_MSE'] for k in range(3)]
    passed=sum(seed_better)>=2 and joint_ci[0]>0
    out=dict(passed=passed,local_complete_beats_eta_state_eta_MSE_by_seed=seed_better,
        aggregate_skill_CI95=joint_ci,aggregate_half_CI95=half_ci,
        rule='>=2/3seeds SOURCEVALinteractionMSE betterthaneta AND 20kstratifiedphysicalstatebootstrap mean-scene interactionskillCIpositive.',
        boundary='SOURCEVAL checkpointselected usingthese labels,so gateisfitness/powerdiagnostic,NOTconfirmation. Heldlabelsnotused. Actiontarget uses everyactivecomponent; rawactionsinclude laternumericalQfailures.',
        next='Runallfourheldlocal/return contrasts asfrozen.' if passed else 'PRUNE remaining72LOSO fits: localtasknotvalidated. Cannotattribute failure specificallytolonghorizon.',
        new_rollout=0,new_query=0)
    write(HERE/'source_gate.json',out);write(HERE/'source_results.json',dict(rows=rows,source_metrics=source_metrics,gate=out))
    np.savez_compressed(HERE/'source_bootstrap.npz',**boots)
    np.savez_compressed(HERE/'source_predictions.npz',**saved)
    lines=['FULL-VECTOR LOCAL-EXECUTION SOURCE GATE',json.dumps(out,sort_keys=True),
        'AllopenedDEVELOPMENT; noheldlabelsusedforgate. Sharedencoderinitializationverified; target-specificoutput/checkpointcriteriadiffer.',
        'scene arm interaction skill CI95']
    lines += [f"{r['scene']} {r['arm']} {r['interaction']} {r['skill']:.7g} {r['CI95']}" for r in rows]
    (HERE/'SOURCE_REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(out,sort_keys=True))
    for r in rows:
        if r['arm']=='complete':print(r['scene'],r['interaction'],r['skill'],r['CI95'])

if __name__=='__main__':run()

