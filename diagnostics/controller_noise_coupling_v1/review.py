"""Paired physical-state evaluation on noise excluded from all new TRAIN labels."""
import argparse
import numpy as np
from .study import HERE,BASE,SEEDS,PAIRS,read,write,sha
from diagnostics.controller_local_to_return_probe_v1.vector_review import center,ci

def run():
    d=dict(np.load(HERE/'dataset.npz'));raw=dict(np.load(BASE/'motion_raw.npz'))
    allp={};fits=[];cpu_pairs=[]
    for j in range(6):
        for s in SEEDS:
            pair=[]
            for arm in ('common','independent'):
                f=HERE/'fits/joint'/f'motion_pair{j}_{arm}_complete'/f'seed{s}'
                q=np.load(f/'predictions.npz');np.testing.assert_array_equal(q['state_uid'],d['state_uid'])
                allp[j,s,arm]=q['probability'][:,[0,5,10,15]].astype(np.float64)
                fits.append(dict(panel=j,coupling=arm,**read(f/'complete.json')))
                pair.append(read(f/'sampling_audit.json')['cpu_affinity'])
            assert pair[0]==pair[1]
            cpu_pairs.append(dict(panel=j,seed=s,affinity=pair[0]))
    ensemble={a:np.mean([allp[j,s,a] for j in range(6) for s in SEEDS],0) for a in ('common','independent')}
    rows=[];per_panel=[];bootout={};saved={}
    for sc in sorted(set(d['scene'])):
        for split in ('train','validation'):
            ix=np.flatnonzero((d['scene']==sc)&(d['split']==split))
            ix=np.array(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
            n=len(ix)//4;na=int(d['entity_agent_mask'][ix[0]].sum())*2;nt=2 if split=='train' else 4
            y=raw['vector_motion'][ix][:,[0,5,10,15],:nt,:na].reshape(n,4,4,nt,na).mean(3)
            pp={a:p[ix,:,:na].reshape(n,4,4,na) for a,p in ensemble.items()}
            tag=str(sc)+'__'+split;saved[tag+'__states']=d['state_uid'][ix][::4];saved[tag+'__truth']=y
            for a,p in pp.items():saved[tag+'__'+a]=p
            ib=np.random.default_rng(2026101034+n).integers(n,size=(20000,n));bootout[tag+'__indices']=ib
            for name,axes in (('state_eta',(0,2)),('state_controller_eta',(0,1,2))):
                pc={a:center(p,axes) for a,p in pp.items()}
                tt=center(y,axes)
                bs={a:[] for a in ('common','independent','common_minus_independent')}
                for start in range(0,len(ib),500):
                    jj=ib[start:start+500];red=(1,2,3,4)
                    cp={a:center(p[jj],tuple(x+1 for x in axes)) for a,p in pp.items()}
                    skills={a:np.mean(2*v*y[jj]-v*v,axis=red) for a,v in cp.items()}
                    for a in ('common','independent'):bs[a].append(skills[a])
                    bs['common_minus_independent'].append(skills['common']-skills['independent'])
                point={a:float(np.mean(2*p*tt-p*p)) for a,p in pc.items()}
                point['common_minus_independent']=point['common']-point['independent']
                for a,v in bs.items():
                    v=np.concatenate(v);bootout[tag+'__'+name+'__'+a]=v
                    rows.append(dict(scene=str(sc),split=split,N=n,interaction=name,comparison=a,
                        skill_or_difference=point[a],CI95=ci(v),
                        total_action_MSE=float(np.mean((pp[a]-y)**2)) if a in pp else None))
                for j in range(6):
                    local={}
                    for arm in ('common','independent'):
                        p=np.mean([allp[j,s,arm] for s in SEEDS],0)[ix,:,:na].reshape(n,4,4,na)
                        p=center(p,axes);local[arm]=float(np.mean(2*p*tt-p*p))
                    per_panel.append(dict(scene=str(sc),split=split,N=n,panel=j,key_subset=PAIRS[j],interaction=name,
                        **local,difference=local['common']-local['independent']))
    failed=[]
    for sc in ('double_bottleneck','four_way_intersection','ring_exchange'):
        for comp in ('common','common_minus_independent'):
            r=next(x for x in rows if x['scene']==sc and x['split']=='train' and x['interaction']=='state_eta' and x['comparison']==comp)
            if r['CI95'][0]<=0:failed.append(dict(scene=sc,condition=comp,CI95=r['CI95']))
        r=next(x for x in rows if x['scene']==sc and x['split']=='validation' and x['interaction']=='state_eta' and x['comparison']=='common_minus_independent')
        if r['CI95'][1]<0:failed.append(dict(scene=sc,condition='SOURCEVALclear_reversal',CI95=r['CI95']))
    result=dict(rows=rows,per_noise_panel=per_panel,gate_passed=not failed,gate_failures=failed,
        cpu_pairs=cpu_pairs,all_paired_CPU_identical=True,unique_cached_actions=10240,new_rollout=0,new_query=0,
        new_fits=36,aggregate_training_labels_identical=True,
        boundary='LOCALaction learning only. TRAIN rows are old physicalstates with noise excluded from these fits, not freshstates. VAL usedforselection. No Qsuccess/LOSO repair claim; anyfuturepositiveclaimrequiresuntouchedstates andnoise. Allsixnoise panelsandthreefitseeds retained.')
    write(HERE/'results.json',result);np.savez_compressed(HERE/'bootstrap.npz',**bootout);np.savez_compressed(HERE/'predictions.npz',**saved)
    lines=['CROSS-STATE NOISE COUPLING: MATCHED LOCAL-LEARNING INTERVENTION',result['boundary'],
        'Paired input/network/budget/CPU/validation fixed. Sixpanels ensureidenticalaggregateTRAINlabels andidenticaluniquedata.',
        f"Gate passed: {not failed}. Failures: {failed}",'scene split comparison state_eta_skill CI95']
    for r in rows:
        if r['interaction']=='state_eta':lines.append(f"{r['scene']} {r['split']} {r['comparison']} {r['skill_or_difference']:.7g} {r['CI95']}")
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    write(HERE/'completion.json',dict(status='COMPLETE_LOCAL_CAUSAL_DIAGNOSTIC',gate_passed=not failed,
        root_cause_closed=False,new_rollout=0,new_query=0,new_fit=36,report_sha256=sha(HERE/'REPORT.txt')))
    print(dict(gate_passed=not failed,gate_failures=failed))
    for r in rows:
        if r['interaction']=='state_eta' and r['comparison']=='common_minus_independent':
            print(r['scene'],r['split'],r['skill_or_difference'],r['CI95'])

if __name__=='__main__':run()

