"""Reuse goal/Flow pilot outcomes; audit all common-program pilot cells."""
import itertools
import numpy as np
from .study import HERE,read,write,FAMILIES
from shared_rollout_db.src.rollout_db import connect
from diagnostics.controller_flow_endpoint_probe_v1.analyze import AXES,GROUPS,energy


def run():
    p=read(HERE/'protocol.json');scenes=sorted(p['controllers']);refs=('goal','flow','navigation')
    prior=HERE.parent/'controller_flow_endpoint_probe_v1';oldp=read(prior/'protocol.json');old=dict(np.load(prior/'raw_labels.npz'))
    for state in p['states']:
        other=next(s for s in oldp['states'] if s['scenario']==state['scenario'] and s['study_index']==state['study_index'])
        assert state['state_uid']==other['state_uid'] and state['physical']==other['physical']
    np.testing.assert_array_equal(old['eta'],np.asarray(p['eta'])[p['eta_indices']])
    shape=(4,8,3,3,4,2);fields={}
    for key in ('success','unknown','collision','episode_length','termination','rollout_uid'):
        array=np.empty(shape,dtype=old[key].dtype)
        for si,scene in enumerate(scenes):
            osi=list(old['scene']).index(scene)
            for ci,family in enumerate(FAMILIES):
                oci=list(old['controller']).index(family)
                array[si,:,:2,ci]=old[key][osi,:,:,oci]
        fields[key]=array
    seen=np.zeros((4,8,3,4,2),bool);etaidx={j:k for k,j in enumerate(p['eta_indices'])}
    with connect(True) as db:
        for c in read(HERE/'cases.json'):
            r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            assert r is not None and not r['conflict_quarantined']
            ix=(scenes.index(c['scene']),c['state_index'],FAMILIES.index(c['controller_name'].removeprefix('navigation_')),etaidx[c['eta_index']],c['seed'])
            assert not seen[ix];seen[ix]=True;ii=ix[:2]+(2,)+ix[2:]
            for key,dbkey in [('success','success'),('unknown','numerical_failure'),('collision','collision'),('episode_length','episode_length'),('termination','outcome'),('rollout_uid','rollout_uid')]:
                fields[key][ii]=r[dbkey]
    assert seen.all()
    np.savez_compressed(HERE/'raw_labels.npz',**fields,scene=np.array(scenes),reference=np.array(refs),controller=np.array(FAMILIES),eta=old['eta'])
    boot=np.random.default_rng(2026101212).integers(8,size=(20000,8));rows=[];effects=[];arrays={}
    for si,scene in enumerate(scenes):
        for ri,ref in enumerate(refs):
            lo=fields['success'][si,:,ri].astype(float);hi=lo+fields['unknown'][si,:,ri]
            sel=lo[...,0].argmax(-1);fixed=np.broadcast_to(lo[...,0].mean(0).argmax(-1),(8,3))
            take=lambda a,s:np.take_along_axis(a,s[...,None],-1)[...,0]
            dl=np.where(sel==fixed,0,take(lo[...,1],sel)-take(hi[...,1],fixed)).mean(-1)
            dh=np.where(sel==fixed,0,take(hi[...,1],sel)-take(lo[...,1],fixed)).mean(-1)
            pairs=[]
            for ci in range(3):
                for j,k in itertools.combinations(range(4),2):
                    pos=lo[:,ci,j,0]-hi[:,ci,k,0]>=1;neg=lo[:,ci,k,0]-hi[:,ci,j,0]>=1
                    if min(pos.sum(),neg.sum())<2:continue
                    da=lo[pos,ci,j,1]-hi[pos,ci,k,1];db=lo[neg,ci,k,1]-hi[neg,ci,j,1]
                    pairs.append(dict(controller=FAMILIES[ci],eta_pair=[p['eta_indices'][j],p['eta_indices'][k]],
                        positive=int(pos.sum()),negative=int(neg.sum()),evaluation_positive_lower=float(da.mean()),
                        evaluation_negative_lower=float(db.mean()),both_directions_confirmed=bool(da.mean()>=.5 and db.mean()>=.5)))
            terms=fields['termination'][si,:,ri]
            rows.append(dict(scene=scene,reference=ref,N=8,rollouts=lo.size,success=int(lo.sum()),
                unknown=int(fields['unknown'][si,:,ri].sum()),collision=int(fields['collision'][si,:,ri].sum()),
                termination={name:int(np.sum(terms==name)) for name in np.unique(terms)},
                success_bounds=[float(lo.mean()),float(hi.mean())],
                crossfit_selection_gain_bounds=[float(dl.mean()),float(dh.mean())],
                crossfit_selection_gain_censor_CI95=[float(np.quantile(dl[boot].mean(1),.025)),float(np.quantile(dh[boot].mean(1),.975))],
                per_controller_eta_success_lower=lo.mean((0,3)).tolist(),
                qualifying_reversal_pairs=len(pairs),independently_confirmed_reversal_pairs=sum(z['both_directions_confirmed'] for z in pairs),reversal_cases=pairs))
            for group in GROUPS:
                axes=[AXES[k] for k in group];pt,l,u=energy(lo,hi,axes);bs=[];bl=[];bh=[]
                for start in range(0,len(boot),1000):
                    ii=boot[start:start+1000];a,b,c=energy(lo[ii],hi[ii],[k+1 for k in axes]);bs.append(a);bl.append(b);bh.append(c)
                bs,bl,bh=map(np.concatenate,(bs,bl,bh));key=scene+'_'+ref+'_'+'x'.join(group)
                arrays[key],arrays[key+'_lower'],arrays[key+'_upper']=bs,bl,bh
                effects.append(dict(scene=scene,reference=ref,centered_over=list(group),independent_half_energy_midpoint=float(pt),
                    identification_bounds=[float(l),float(u)],censor_CI95=[float(np.quantile(bl,.025)),float(np.quantile(bh,.975))]))
        for oldref in (0,1):
            lo=(fields['success'][si,:,2]-fields['success'][si,:,oldref]-fields['unknown'][si,:,oldref]).mean((1,2,3))
            hi=(fields['success'][si,:,2]+fields['unknown'][si,:,2]-fields['success'][si,:,oldref]).mean((1,2,3))
            rows.append(dict(scene=scene,reference='navigation_minus_'+refs[oldref],N=8,success_difference_bounds=[float(lo.mean()),float(hi.mean())],
                censor_CI95=[float(np.quantile(lo[boot].mean(1),.025)),float(np.quantile(hi[boot].mean(1),.975))]))
    write(HERE/'data_quality_results.json',dict(rows=rows,signal=effects,new_outcomes=768,reused_outcomes=1536,
        interpretation='DEVELOPMENT pilot, no learned critic. Independent-half covariance estimates reproducible variation. Correct map-aware program is shared acrossscenes; source/target geometry and physicalenvironment constants still differ.',
        limitations='8naturalTRAINstates per scene,K4,Q2; low-power reversalcheck. No final confirmation or B15 claim. Comparisons change the whole baseprogram/noiselaw, not a pure single-feature intervention.'))
    np.savez_compressed(HERE/'signal_bootstrap.npz',**arrays)
    for r in effects:
        if r['reference']=='navigation' and r['centered_over']==['state','eta']:print(r,flush=True)
    for r in rows:
        if r['reference']=='navigation':print({k:v for k,v in r.items() if k not in ['reversal_cases','per_controller_eta_success_lower']},flush=True)


if __name__=='__main__':run()
