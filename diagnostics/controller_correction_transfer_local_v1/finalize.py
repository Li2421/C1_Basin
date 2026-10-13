"""All-fold deterministic lift review and fixed eta-basis error localization."""
import csv
import numpy as np
from .study import HERE,FOLDS,SEEDS,read,write,sha


def run():
    d=np.load(HERE/'dataset.npz');p=read(HERE/'protocol.json')
    t=np.load(HERE/'goal_frame_basis_truth.npz');lookup={s:b for s,b in zip(t['state_uid'],t['basis'])}
    design=np.column_stack((np.ones(4),d['eta'][[0,5,10,15]]));inv=np.linalg.inv(design)
    basis_rows=[];boots={};rows=[]
    for fold,held in FOLDS.items():
        rr=read(HERE/f'review_{fold}.json')['rows'];rows.extend(rr)
        pred=np.mean([np.load(HERE/'fits'/fold/'motion_complete'/f'seed{s}'/'predictions.npz')['probability'].astype(float) for s in SEEDS],0)
        groups=[(sc,'source_validation',d['split']=='validation') for sc in FOLDS.values() if sc!=held]
        groups.append((held,'held_all',np.ones(len(d['scene']),bool)))
        oldboot=np.load(HERE/f'bootstrap_{fold}.npz')
        for sc,name,mask in groups:
            ii=np.flatnonzero((d['scene']==sc)&mask);ii=np.array(sorted(ii,key=lambda i:(d['state_uid'][i],d['controller'][i])))
            sid=d['state_uid'][ii][::4];n=len(sid);na=int(d['entity_agent_mask'][ii[0]].sum());no=2*na
            y=pred[ii][:,[0,5,10,15],:no].reshape(n,4,4,no)
            coef=np.einsum('ki,ncij->nckj',inv,(2*y-1)*p['correction_bound_vmax'])
            basis=np.asarray([lookup[s][:na] for s in sid]).reshape(n,no,3).transpose(0,2,1)
            truth=np.concatenate((np.zeros((n,1,no)),basis),1)
            tag=str(sc)+'__'+name;ib=oldboot[tag+'__indices'];w=np.array([np.bincount(z,minlength=n) for z in ib],float)/n
            for j,label in enumerate(('intercept_expected_zero','B_goal','B_safe_perp','B_relative')):
                pp=coef[:,:,j,:].reshape(n,-1);tt=np.repeat(truth[:,None,j,:],4,1).reshape(n,-1)
                delta=pp-tt;pc=pp-pp.mean(0);tc=tt-tt.mean(0)
                initial=np.mean(2*pp*tt-pp*pp,1);mp=w@pp;mt=w@tt
                sb=w@initial-np.mean(2*mp*mt-mp*mp,1)
                eb=w@np.mean(delta*delta,1)
                key='__'.join((fold,tag,label));boots[key+'__state_skill']=sb;boots[key+'__MSE']=eb
                basis_rows.append(dict(fold=fold,scene=sc,split=name,physical_states=n,basis=label,
                    MSE=float(np.mean(delta*delta)),MSE_CI95=np.quantile(eb,[.025,.975]).tolist(),
                    state_effect_skill=float(np.mean(2*pc*tc-pc*pc)),state_skill_CI95=np.quantile(sb,[.025,.975]).tolist(),
                    true_state_energy=float(np.mean(tc*tc)),
                    spurious_controller_variance=float(np.mean((coef[:,:,j]-coef[:,:,j].mean(1,keepdims=True))**2))))
    held=[r for r in rows if r['split']=='held_all' and r['metric']=='state_eta']
    source=[r for r in rows if r['split']=='source_validation' and r['metric']=='state_eta']
    positive=[r['fold'] for r in held if r['skill_CI95'][0]>0]
    result=dict(rows=rows,basis_rows=basis_rows,held_positive_folds=positive,
        source_positive_count=sum(r['skill_CI95'][0]>0 for r in source),source_checks=len(source),
        new_fits=24,new_rollouts=0,new_FF_actions=0,new_basis_queries=128,reused_basis_states=32,
        root_cause_closed=False,deterministic_target=True,
        interpretation='Actualdelta target precedes FF integration and is identicalacrossfourMfamilies. Tripleinteraction shouldbezero; negative triple skill here diagnoses spuriousCdependence,notfailuretolearnrealCinteraction. Allhalfpower outputs meanexactdeterministictargetenergy,notindependentrolloutreliability.',
        next_decision='Delta transfers allfour: moveforwardoneoperation towardFFcomposition.' if len(positive)==4 else
        'Delta transfer not universal; inspect source learning and basis decomposition to locate upstream physical/intervention learning failure. Do not explain this deterministicdeltafailure by hiddenfutureFlow or rolloutnoise; stillnotproofthisalonecausesalllongQfailure.')
    write(HERE/'results.json',result);np.savez_compressed(HERE/'basis_bootstrap.npz',**boots)
    for file,data in [('summary.csv',rows),('basis_summary.csv',basis_rows)]:
        with (HERE/file).open('w') as f:
            keys=sorted(set().union(*(r.keys() for r in data)));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
    lines=['DETERMINISTIC ACTUAL CORRECTION TRANSFER — BEFORE FF INTEGRATION',
        f'Source state_eta positiveCI: {result["source_positive_count"]}/{len(source)}; heldpositive: {positive}',
        result['next_decision'],result['interpretation'],'',
        'Scene | state_eta skill [95%CI] | exact energy | eachfitseed']
    for r in held:lines.append(f'{r["scene"]} | {r["interaction_skill"]:.8g} {r["skill_CI95"]} | {r["true_interaction_energy"]:.8g} | {r["per_seed_skill"]}')
    lines.extend(['','FIXED ETA-BASIS INVERSION (all4coefficients, all4candidates; condition~6.01):'])
    for r in basis_rows:
        if r['split']=='held_all':lines.append(f'{r["scene"]} {r["basis"]}: state_skill {r["state_effect_skill"]:.8g} CI{r["state_skill_CI95"]}; trueenergy{r["true_state_energy"]:.8g}; MSE{r["MSE"]:.8g}')
    lines.extend(['','Inputaudit: Bgoal/Brel reconstructed directlyfromexistingphysicalfeatures to1e-6 tolerance; notuniversalQ/controller sufficiency.',
        'All160 oldnaturalstates/K4/fourgoalM reused;72input/features/network/sampling6000steps/sourceVALrule unchanged. Actualdelta depends on preparatory safety(nominalreference), so notyetpuregeometry-onlylocalization.',
        '32oldbasisstates reused,128missingbasisfunctionqueries. No FFintegration,no physicalsteps,nocollision/success/terminalQclaims. Rawtargets/worldbasis/featurehashes/sourcecheckpointIDs/predictions/statebootstrap saved.',
        'Four identical targetcopies are arrayformatcompatibility only. Bootstrapunitphysicalstate; no rolloutpseudo-replication. Fullvector and state-bundle mismatch, Cvariation asnegativecontrol. OpenedDEVELOPMENT, no finalconfirmation.',
        'Primary literature consulted: https://arxiv.org/abs/1806.01261; structural/compositional learning motivation, no theoremforoursystem.'])
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    write(HERE/'completion.json',dict(status='COMPLETE_DETERMINISTIC_CORRECTION_LOCALIZATION',root_cause_closed=False,
        held_positive_folds=positive,new_rollouts=0,report_sha256=sha(HERE/'REPORT.txt')))
    print(dict(source_positive=result['source_positive_count'],held_positive=positive))
    for r in held:print(r['fold'],r['skill_CI95'])
    for r in basis_rows:
        if r['split']=='held_all' and r['basis']=='B_relative':print(r['fold'],'Brelative',r['state_skill_CI95'])


if __name__=='__main__':run()
