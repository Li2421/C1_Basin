"""Source-only inputs, mandatory descriptor/guard/prior checks and deduplicated fits."""
from common import *
from descriptors_v2 import descriptor,EXPECTED,ranges,target_guard
from anchored import prior
import numpy as np

V=ROOT/'diagnostics/cross_scene_pipeline_v1';O=ROOT/'diagnostics/cross_scene_correction_v1'

def halt(reason,**details):
    write(HERE/'STOP.receipt.json',dict(utc=utc(),reason=reason,**details));raise RuntimeError(reason)

def normalization(inputs,sources,field):
    hh=np.vstack([np.asarray(inputs[n]['h'])[inputs[n]['training_state_positions']]for n in sources])
    cc=np.vstack([np.repeat(np.asarray(inputs[n][field])[None],len(inputs[n]['training_state_positions']),axis=0)for n in sources])
    hs=hh.std(0);cs=cc.std(0)
    return dict(h_mean=hh.mean(0).tolist(),h_scale=np.where(hs<1e-6,1,hs).tolist(),c_mean=cc.mean(0).tolist(),c_scale=np.where(cs<1e-6,1,cs).tolist(),
        fit_scenes=sources,fit_state_indices={n:[inputs[n]['state_indices'][i]for i in inputs[n]['training_state_positions']]for n in sources})

def main():
    assert read(HERE/'REPLAY_CHECK.json')['passed']and read(HERE/'REQUEST_RECONSTRUCTION_CHECK.json')['passed']
    assert(HERE/'V1_DIAGNOSIS.json').exists()
    old=read_evidence(V/'INPUTS.json');tables=read_evidence(O/'LABEL_TABLES.json');scene_set=read_evidence(V/'SCENE_SET.json')
    scenes=[s for s in read_evidence(O/'SCENES.json')if s['family']!='extrapolation'];names=[s['name']for s in scenes]
    assert len(names)==18 and set(names)==set(EXPECTED)
    inputs=copy.deepcopy(old['scene_inputs']);checks={}
    for s in scenes:
        n=s['name'];c=descriptor(s);target=np.r_[EXPECTED[n],[1,.5,6]if n=='lit_swap8_moving'else[0,0,0],int(n.startswith('xsc_swap_')or n.startswith('lit_swap8_'))]
        delta=np.abs(np.round(c,3)-target);checks[n]=dict(actual=c.tolist(),rounded=np.round(c,3).tolist(),expected=target.tolist(),max_absolute_difference=float(delta.max()))
        if np.any(delta>.002):halt('DESCRIPTOR_TABLE_MISMATCH',scene=n,check=checks[n])
        inputs[n]['c_raw']=inputs[n]['c'];inputs[n]['c_prime']=c.tolist()
    fields=['log人数','log半径/平均行程','log每步位移/半径','log时限/最长直线行程时间','面积占比','墙体连通块数','log最小开口/直径','串联墙数','移动障碍','移动速度比','移动半径比','对径交换']
    write(HERE/'SCENE_SET.json',scene_set)
    write(HERE/'INPUTS_V2.json',dict(scene_inputs=inputs,etas=old['etas'],eta_keys=old['eta_keys'],controller_order=old['controller_order'],state_fields=old['state_fields'],
        description_fields=fields,descriptor_checks=checks,labels_not_in_input_features=True,guard_margin=.25,development_after_v1_results=True))
    # Compatibility read object for copied utilities; no v1 module imports.
    write(HERE/'INPUTS.json',dict(old,scene_inputs=inputs))
    etas=np.asarray(old['etas'],float);outer=read_evidence(V/'FOLDS.json');guard_rows={};outer_priors={}
    for f in outer:
        n=f['name'];folder=V/'folds'/n;baseline=read_evidence(folder/'definition.json')['baselines']
        bounds=ranges(inputs,f['sources']);raw_bounds=ranges(inputs,f['sources'],'c_raw')
        guard_rows[n]={t:dict(scale_independent=target_guard(inputs,t,bounds),raw=target_guard(inputs,t,raw_bounds,'c_raw'))for t in f['targets']}
        p=prior(tables,f['sources'],etas);outer_priors[n]=p
        for t in f['targets']:
            if old['eta_keys'][p['best_index']]!=baseline[t]['global_fixed']:
                halt('PRIOR_BASELINE_MISMATCH',fold=n,scene=t,actual=old['eta_keys'][p['best_index']],expected=baseline[t]['global_fixed'])
        f['baselines']=baseline
    expected_loso={'lit_narrow_passage','lit_circle_five','xsc_swap_n12'}
    expected_full={'lit_narrow_passage','lit_circle_five','lit_swap8_moving','xsc_gap_n2'}
    expected_lofo={'lofo__bottleneck':{'xsc_gap_n4','xsc_gap_n6','xsc_double_n4','xsc_offset_n4'},'lofo__literature':{'lit_narrow_passage','lit_crowded_random','lit_circle_five'}}
    actual={kind:{t for f in outer if f['kind']==kind for t in f['targets']if guard_rows[f['name']][t]['scale_independent']['scene_out_of_range']}for kind in ('loso','full18')}
    if actual['loso']!=expected_loso or actual['full18']!=expected_full:halt('SCENE_GUARD_SET_MISMATCH',actual={k:sorted(v)for k,v in actual.items()})
    for n,expected in expected_lofo.items():
        got={t for t,g in guard_rows[n].items()if g['scale_independent']['scene_out_of_range']}
        if got!=expected:halt('LOFO_GUARD_SET_MISMATCH',fold=n,actual=sorted(got),expected=sorted(expected))
    by_loso={t:g['scale_independent']['scene_trigger_dimensions']for f in outer if f['kind']=='loso'for t,g in guard_rows[f['name']].items()if g['scale_independent']['scene_out_of_range']}
    if by_loso!={'lit_narrow_passage':[1,4],'lit_circle_five':[3,5],'xsc_swap_n12':[1]}:halt('GUARD_DIMENSION_MISMATCH',actual=by_loso)
    write(HERE/'GUARD.json',dict(passed=True,folds=guard_rows,expected_sets_verified=True,expected_loso_dimensions_verified=True,state_ranges_use_all_source_states=True))
    tasks_inner=[];tasks_outer=[];inner_nodes={};nodes={}
    def definition(name,sources,targets,field,kind='inner',baselines=None):
        norm=normalization(inputs,sources,field);arrays={}
        for s in sources:
            d=inputs[s];arrays.update({s+'__h':np.asarray(d['h']),s+'__c':np.repeat(np.asarray(d[field])[None],len(d['h']),axis=0),
                s+'__onehot':np.asarray(d['controller_one_hot']),s+'__y':np.asarray(tables[s]['y'],float),s+'__state_indices':np.asarray(d['state_indices']),
                s+'__train':np.asarray(d['training_state_positions']),s+'__val':np.asarray(d['validation_state_positions'])})
        folder=HERE/'folds'/name;folder.mkdir(parents=True,exist_ok=True);label=folder/('source_labels_raw.npz'if field=='c_raw'else'source_labels.npz');np.savez_compressed(label,**arrays)
        out=dict(name=name,kind=kind,sources=sources,targets=targets,descriptor=field,normalization=norm,source_labels_path=str(label),training_data_sha256=sha(label),
            baselines=baselines or{},generator=kind!='full18',modes=['full'],ranges=ranges(inputs,sources,field))
        filename='raw_definition.json'if field=='c_raw'and kind!='inner'else'definition.json'
        write(folder/filename,out);return out
    for f in outer:
        n=f['name'];sources=f['sources'];main=definition(n,sources,f['targets'],'c_prime',f['kind'],f['baselines']);main['generator']=f['generator']
        write(HERE/'folds'/n/'definition.json',main);write(HERE/'folds'/n/'PRIORS.json',outer_priors[n]);nodes[n]=main
        raw=None
        if f['generator']:raw=definition(n,sources,f['targets'],'c_raw',f['kind'],f['baselines'])
        for mode in ['full']+(['raw','no_prior','generator']if f['generator']else[]):
            for seed in (17,23,41):tasks_outer.append(dict(fold=n,kind='generator'if mode=='generator'else'critic',mode=mode,seed=seed,source_scene_count=len(sources)))
        links={}
        for mode,field in [('full','c_prime')]+([('raw','c_raw')]if f['generator']else[]):
            links[mode]={}
            for target in sources:
                subset=[s for s in sources if s!=target]
                key=uid('inner',dict(sources=subset,descriptor=field,seed=17,kind='anchored_critic'))
                name=key
                if key not in inner_nodes:
                    node=definition(name,subset,[s for s in names if s not in subset],field)
                    p=prior(tables,subset,etas);write(HERE/'folds'/name/'PRIORS.json',p);inner_nodes[key]=node
                    tasks_inner.append(dict(fold=name,kind='critic',mode=mode,seed=17,source_scene_count=len(subset)))
                node=inner_nodes[key]
                links[mode][target]=dict(training_node=name,mode=mode,target=target,guard=target_guard(inputs,target,node['ranges'],field))
        main['inner_links']=links;write(HERE/'folds'/n/'definition.json',main);nodes[n]=main
    write(HERE/'FOLDS.json',list(nodes.values()));write(HERE/'INNER_NODES.json',list(inner_nodes.values()))
    write(HERE/'TRAIN_INNER_TASKS.json',tasks_inner);write(HERE/'TRAIN_OUTER_TASKS.json',tasks_outer)
    counts=dict(planned_unshared_inner_fits=554,unique_inner_fits=len(tasks_inner),outer_fits=len(tasks_outer),total_unique_fits=len(tasks_inner)+len(tasks_outer),
        source_identical_inner_fits_reused=554-len(tasks_inner),criterion='Exact ordered sources, descriptor, architecture/loss/sampling and seed17; same priors, normalization and source labels. No target labels in fitting.')
    write(HERE/'TRAINING_PLAN.json',counts)
    write(HERE/'INPUTS_V2.receipt.json',dict(utc=utc(),passed=True,descriptor_table_passed=True,guard_sets_passed=True,all_outer_prior_maxima_equal_v1=True,
        files={str(HERE/n):sha(HERE/n)for n in ('INPUTS_V2.json','INPUTS.json','SCENE_SET.json','GUARD.json','FOLDS.json','INNER_NODES.json','TRAIN_INNER_TASKS.json','TRAIN_OUTER_TASKS.json')}))
    print('INPUTS V2 verified.',counts,flush=True)

if __name__=='__main__':main()
