"""Source-type scaling plus feasible controls; never fabricate extra instances."""
from common import *
from development import load,TYPES
from selector import Regression,tune
from selector_candidates_v2 import fit_selected
from analysis_statistics import sample_indices,point_indices,oracle_choices,gather_arm
from html import escape


def interval(data,choices,replicates=5000):
    types=data['types'];labels=data['labels'];rng=np.random.default_rng(202610180501)
    valid=choices>=0;assert valid.any(1).all()
    repetition_weight=valid/valid.sum(1)[:,None]
    def calculate(cc,ss,weights):
        sampled=labels[cc[...,None],ss];lo=(sampled==1).astype(float);hi=(sampled!=0).astype(float)
        oracle=oracle_choices(lo,ss,types[cc],weights)['crossfit_oracle'];ol=gather_arm(lo,oracle);oh=gather_arm(hi,oracle)
        point=np.zeros_like(ol);lower=np.zeros_like(ol);upper=np.zeros_like(ol)
        for r in range(choices.shape[1]):
            selected=np.broadcast_to(np.maximum(choices[cc,r],0)[...,None],ss.shape);sl=gather_arm(lo,selected);sh=gather_arm(hi,selected)
            rw=repetition_weight[cc,r][...,None,None]
            same=(selected==oracle)[...,None];point+=(ol-sl)*rw;lower+=np.where(same,0.,ol-sh)*rw;upper+=np.where(same,0.,oh-sl)*rw
        controller=np.stack([a.mean((-1,-2)) for a in (point,lower,upper)],axis=-1)
        return np.sum(controller*weights[...,None],axis=1)
    cc,ss,weights=point_indices(types,labels.shape[1],True);point=calculate(cc,ss,weights)[0]
    draws=[]
    for start in range(0,replicates,64):
        cc,ss,weights=sample_indices(types,labels.shape[1],rng,min(64,replicates-start),True)
        draws.append(calculate(cc,ss,weights))
    draws=np.concatenate(draws)
    return dict(regret_pp=float(point[0]*100),unknown_bounds_pp=(point[1:]*100).tolist(),
        CI95_pp=[float(np.quantile(draws[:,1],.025)*100),float(np.quantile(draws[:,2],.975)*100)],
        replicates=replicates,source_group_repetitions_averaged=True,oracle_reselected_within_bootstrap=True)


def paired_gain(data,a,b,replicates=5000):
    """Paired contrast of two choice mixtures, cancelling shared unknown cases."""
    from selector import ARMS
    def frequencies(choices):
        valid=choices>=0;assert valid.any(1).all()
        return np.stack([(choices==j).sum(1)/valid.sum(1) for j in range(len(ARMS))],axis=1)
    delta=frequencies(a)-frequencies(b);types=data['types'];labels=data['labels']
    def calculate(cc,ss,weights):
        values=labels[cc[...,None],ss];lo=(values==1).astype(float);hi=(values!=0).astype(float)
        coefficient=delta[cc][...,None,:,None]
        point=np.sum(coefficient*lo,axis=-2)
        lower=np.sum(np.where(coefficient>=0,coefficient*lo,coefficient*hi),axis=-2)
        upper=np.sum(np.where(coefficient>=0,coefficient*hi,coefficient*lo),axis=-2)
        return np.sum(np.stack([v.mean((-1,-2)) for v in (point,lower,upper)],axis=-1)*weights[...,None],axis=1)
    cc,ss,w=point_indices(types,labels.shape[1],True);point=calculate(cc,ss,w)[0]
    rng=np.random.default_rng(202610180501);draws=[]
    for start in range(0,replicates,64):
        cc,ss,w=sample_indices(types,labels.shape[1],rng,min(64,replicates-start),True);draws.append(calculate(cc,ss,w))
    draws=np.concatenate(draws)
    return dict(gain_pp=float(point[0]*100),unknown_bounds_pp=(point[1:]*100).tolist(),
        CI95_pp=[float(np.quantile(draws[:,1],.025)*100),float(np.quantile(draws[:,2],.975)*100)],replicates=replicates)


def fit_choice(data,rates,indices,held,strategy='ridge'):
    indices=np.asarray(indices,int)
    model,inner=fit_selected(data['features'][indices],rates[indices],data['types'][indices],strategy)
    parameters={'penalty':model.penalty} if strategy=='ridge' else model.config
    return model.choose(data['features'][held]),parameters,inner


def figure(main,supplement,control,path):
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="530" viewBox="0 0 1200 530">',
         '<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:DejaVu Sans,Arial,sans-serif}</style>']
    def text(x,y,t,size=14):svg.append(f'<text x="{x}" y="{y}" font-size="{size}">{escape(str(t))}</text>')
    def panel(rows,x0,title,reference=None):
        ymin=min([r['CI95_pp'][0] for r in rows]+([reference['CI95_pp'][0]] if reference else []));ymax=max([r['CI95_pp'][1] for r in rows]+([reference['CI95_pp'][1]] if reference else []))
        pad=max((ymax-ymin)*.1,.1);ymin-=pad;ymax+=pad
        px=lambda m:x0+65+(m-1)*68;py=lambda v:415-(v-ymin)/(ymax-ymin)*310
        text(x0+45,40,title,17);points=[]
        for v in np.linspace(ymin,ymax,6):
            yy=py(v);svg.append(f'<line x1="{x0+65}" y1="{yy}" x2="{x0+475}" y2="{yy}" stroke="#ddd"/>');text(x0+15,yy+5,f'{v:.2f}',12)
        if reference:
            top=py(reference['CI95_pp'][1]);bottom=py(reference['CI95_pp'][0])
            svg.append(f'<rect x="{x0+65}" y="{top}" width="410" height="{bottom-top}" fill="#888" fill-opacity=".12"/>')
            yy=py(reference['regret_pp']);svg.append(f'<line x1="{x0+65}" y1="{yy}" x2="{x0+475}" y2="{yy}" stroke="#888" stroke-width="2" stroke-dasharray="6 4"/>')
            text(x0+65,78,'Dashed: seven instances from one Flow source type',11)
        for row in rows:
            x,y=px(row['m']),py(row['regret_pp']);points.append(f'{x},{y}')
            svg.append(f'<line x1="{x}" y1="{py(row["CI95_pp"][0])}" x2="{x}" y2="{py(row["CI95_pp"][1])}" stroke="#1a659e" stroke-width="2"/>')
            svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#1a659e"/>');text(x-4,440,row['m'])
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#1a659e" stroke-width="2"/>')
        text(x0+150,475,'Number of source controller types');text(x0+65,500,'Type-equal cross-fitted oracle regret (percentage points)',11)
    panel(main,0,'All available instances per sampled source type')
    panel(supplement,600,'Fixed seven instances; seven feasible held types',control)
    svg.append('</svg>');path.write_text('\n'.join(svg)+'\n')


def main(bundle,iteration):
    data=load(bundle);types=data['types'];rates=(data['labels']==1).mean((1,3));nc=len(types)
    assert set(types)==set(TYPES) and all(np.sum(types==t)>=4 for t in TYPES)
    result=read(HERE/'iterations'/iteration/'RESULTS.json');assert result['input_sha256']==sha(bundle)
    strategy=result.get('strategy','ridge')
    folder=HERE/'scale';assert not folder.exists();folder.mkdir()
    rng=np.random.default_rng(202610180701);control_rng=np.random.default_rng(202610180702)
    rows=[];main_choices={};matched=[]
    for m in (1,2,3,5,7):
        main_choices[m]=np.full((nc,1 if m==7 else 20),-1,int)
    # Preserve the original prespecified held-type/m/repetition RNG order.
    for held_type in TYPES:
        held=types==held_type;source=[t for t in TYPES if t!=held_type]
        for m in (1,2,3,5,7):
            for repetition in range(main_choices[m].shape[1]):
                selected=sorted(rng.choice(source,size=m,replace=False).tolist());indices=np.flatnonzero(np.isin(types,selected))
                chosen,penalty,inner=fit_choice(data,rates,indices,held,strategy);main_choices[m][held,repetition]=chosen
                rows.append(dict(held_type=held_type,m=m,repetition=repetition,source_types=selected,source_controllers=len(indices),
                    source_only_parameters=penalty,inner=inner,held_names=data['names'][held].tolist(),choices=chosen.tolist()))
                # Exact N-matched one-type controls exist only for small N.
                donors=[t for t in source if np.sum(types==t)>=len(indices)]
                if donors:
                    donor=str(control_rng.choice(donors));pool=np.flatnonzero(types==donor)
                    sample=control_rng.choice(pool,len(indices),replace=False);ctl,cp,ci=fit_choice(data,rates,sample,held,strategy)
                    matched.append(dict(held_type=held_type,m=m,repetition=repetition,n=len(indices),available=True,
                        source_type=donor,source_indices=sample.tolist(),choices=ctl.tolist(),source_only_parameters=cp))
                else:matched.append(dict(held_type=held_type,m=m,repetition=repetition,n=len(indices),available=False,
                    reason='No single allowed source type contains this many distinct instances'))
    with np.load(HERE/'iterations'/iteration/'CHOICES.npz') as z:assert np.array_equal(main_choices[7][:,0],z['pipeline'])
    write(folder/'MAIN_TRAINING_RECORDS.json',rows);write(folder/'MATCHED_INSTANCE_CONTROL.json',matched)
    main_curve=[dict(m=m,**interval(data,c)) for m,c in main_choices.items()]
    matched_curve=[]
    for m in (1,2,3,5,7):
        control_choice=np.full_like(main_choices[m],-1);diverse_choice=np.full_like(main_choices[m],-1)
        for row in matched:
            if row['m']!=m or not row['available']:continue
            mask=types==row['held_type'];r=row['repetition']
            control_choice[mask,r]=row['choices'];diverse_choice[mask,r]=main_choices[m][mask,r]
        keep=np.any(control_choice>=0,axis=1)
        if not keep.any():
            matched_curve.append(dict(m=m,available=False,reason='No source type has enough distinct instances'));continue
        sub={k:v[keep] for k,v in data.items() if isinstance(v,np.ndarray) and v.ndim and len(v)==nc and k not in ('arms','feature_names','state_uids')}
        matched_curve.append(dict(m=m,available=True,held_types=sorted(set(types[keep].tolist())),
            feasible_group_counts={str(t):int(np.sum(control_choice[np.flatnonzero(types==t)[0]]>=0)) for t in set(types[keep])},
            diverse=interval(sub,diverse_choice[keep]),single_type=interval(sub,control_choice[keep]),
            paired_gain=paired_gain(sub,diverse_choice[keep],control_choice[keep]),
            interpretation='Conditional on exactly count-matched feasible source groups; held cohorts can differ across m'))
    write(folder/'MATCHED_INSTANCE_CURVE.json',matched_curve)
    # An additional, genuinely fixed-N comparison: seven total instances.
    # Only Flow has >=7 instances, so Flow-held tests have no one-type control.
    eligible=types!='Flow';nsub=int(eligible.sum());supp_choices={m:np.full((nc,20),-1,int) for m in (1,2,3,5,7)}
    control=np.full((nc,20),-1,int);supp_records=[];rng=np.random.default_rng(202610180703)
    for held_type in TYPES:
        if held_type=='Flow':continue
        held=types==held_type;source=[t for t in TYPES if t!=held_type]
        for repetition in range(20):
            single=rng.choice(np.flatnonzero(types=='Flow'),7,replace=False)
            choice,penalty,inner=fit_choice(data,rates,single,held,strategy);control[held,repetition]=choice;supp_choices[1][held,repetition]=choice
            for m in (2,3,5,7):
                selected=sorted(rng.choice(source,m,replace=False).tolist())
                pools={t:list(rng.permutation(np.flatnonzero(types==t))) for t in selected}
                indices=[pools[t].pop() for t in selected]
                while len(indices)<7:
                    donor=str(rng.choice([t for t in selected if pools[t]]));indices.append(pools[donor].pop())
                choice,penalty,inner=fit_choice(data,rates,indices,held,strategy);supp_choices[m][held,repetition]=choice
                supp_records.append(dict(held_type=held_type,m=m,repetition=repetition,source_indices=indices,source_types=selected,
                    source_only_parameters=penalty,choices=choice.tolist(),instances=7))
    subset={k:v[eligible] for k,v in data.items() if isinstance(v,np.ndarray) and v.ndim and len(v)==nc and k not in ('arms','feature_names','state_uids')}
    supplementary=[dict(m=m,**interval(subset,c[eligible])) for m,c in supp_choices.items()]
    single=interval(subset,control[eligible])
    fixed7_comparisons=[dict(m=m,**paired_gain(subset,c[eligible],control[eligible])) for m,c in supp_choices.items()]
    write(folder/'FIXED_SEVEN_RECORDS.json',supp_records)
    output=dict(utc=utc(),iteration=iteration,source_bundle_sha256=sha(bundle),pipeline_structure=strategy,hyperparameter_selection='source-only nested',
        main_curve=main_curve,main_m7_vs_m1_paired_gain=paired_gain(data,main_choices[7],main_choices[1]),supplementary_fixed_seven=supplementary,single_type_seven=single,
        exact_matched_feasible_curve=matched_curve,fixed7_paired_gain_vs_single=fixed7_comparisons,
        full_main_matched_control_available=all(r['available'] for r in matched),
        gaps=['Full N-matched one-type controls are impossible when N exceeds the largest available source type; omitted, never replicated.',
              'The fixed-seven supplement excludes Flow-held tests and uses Flow as its one-type donor; its conclusion is conditional on this feasible cohort.'],
        monotonic_point_estimates=all(a['regret_pp']>=b['regret_pp']-1e-12 for a,b in zip(main_curve,main_curve[1:])),
        final_m7_matches_development_outer_predictions=True)
    write(HERE/'SCALE_CURVE_RESULTS.json',output)
    figure(main_curve,supplementary,single,folder/'scale_curve.svg')
    np.savez_compressed(folder/'CHOICES.npz',**{f'm{m}':c for m,c in main_choices.items()},**{f'fixed7_m{m}':c[eligible] for m,c in supp_choices.items()},fixed7_control=control[eligible])
    seal('SCALE_CURVE_COMPLETE.receipt.json',[HERE/'SCALE_CURVE_RESULTS.json',folder/'MAIN_TRAINING_RECORDS.json',folder/'MATCHED_INSTANCE_CONTROL.json',
        folder/'FIXED_SEVEN_RECORDS.json',folder/'MATCHED_INSTANCE_CURVE.json',folder/'CHOICES.npz',folder/'scale_curve.svg',HERE/'scale_complete.py',HERE/'selector.py',HERE/'selector_candidates.py',HERE/'selector_candidates_v2.py'],
        full_matched_control_has_explicit_gaps=True,no_new_rollouts=True)
    print('Scale curve complete with explicit matched-control feasibility gaps',flush=True)

if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2])
