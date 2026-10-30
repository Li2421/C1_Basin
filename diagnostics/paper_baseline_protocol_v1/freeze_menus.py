"""Prospective paper menus, constructed without rollout outcomes."""
import guard
from guard import HERE
import ast
import copy
import hashlib
import json

BASE = HERE/'inputs/diagnostics'
VENDOR = BASE/'paper_baselines_20261028/vendor'
DEFAULTS = {
    'dac_author_si': dict(alpha=2., agent_threshold=.5, obstacle_threshold=.01),
    'mgr_author_unicycle_tracking': dict(tau=.5, kp=.05, circle_radius=.3),
    'chen_impc_ob_si_tracking': dict(K=8, scale_lengths=False, epsilon_gain=1.),
}
AXES = {
    'dac_author_si': ('alpha','agent_threshold','obstacle_threshold'),
    'mgr_author_unicycle_tracking': ('tau','kp','circle_radius'),
    'chen_impc_ob_si_tracking': ('K','epsilon_gain'),
}


def read(path):
    return json.loads(path.read_text())


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def historical_variants():
    # Extract a literal assignment, never import the old runner or its paths.
    source = BASE/'paper_baselines_20261028/paper_design.py'
    tree = ast.parse(source.read_text())
    freeze = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='freeze')
    assignment = next(n for n in freeze.body if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='variants' for t in n.targets))
    result = {}
    for key,value in zip(assignment.value.keys,assignment.value.values):
        name = ast.literal_eval(key)
        if name not in DEFAULTS:
            continue
        if name=='chen_impc_ob_si_tracking':
            # Literal comprehension axes in the archived assignment.
            assert isinstance(value,ast.ListComp)
            axes=[ast.literal_eval(g.iter) for g in value.generators]
            assert axes==[(8,24),(False,True)]
            result[name]=[dict(K=k,scale_lengths=s) for k in axes[0] for s in axes[1]]
        else:
            assert isinstance(value,ast.List)
            options=[]
            for item in value.elts:
                if isinstance(item,ast.Dict): options.append(ast.literal_eval(item))
                else:
                    assert isinstance(item,ast.Call) and isinstance(item.func,ast.Name) and item.func.id=='dict' and not item.args
                    options.append({kw.arg:ast.literal_eval(kw.value) for kw in item.keywords})
            result[name]=options
    assert set(result)==set(DEFAULTS)
    return result


def normalize(method,options):
    result=copy.deepcopy(DEFAULTS[method])
    options=dict(options)
    profile=options.pop('threshold_profile',None)
    if profile is not None:
        assert method=='dac_author_si' and profile=='author_narrow'
        result.update(agent_threshold=.5,obstacle_threshold=.01)
    result.update(options)
    if 'K' in result:
        result['K']=int(result['K'])
    assert set(result)==set(DEFAULTS[method]),(method,result)
    return result


def provenance():
    specs={
        'dac_author_si': dict(
            alpha=('dac/agents/circle2DAgent.py',47,'CLF alpha class default2'),
            agent_threshold=('dac/sim-narrow_passage.py',45,'author narrow-passage entry Cfg.cbf_boundary_hold agentCA0.5'),
            obstacle_threshold=('dac/sim-narrow_passage.py',45,'author narrow-passage entry Cfg.cbf_boundary_hold obsCA0.01')),
        'mgr_author_unicycle_tracking': dict(
            tau=('mgr/main.py',81,'entry-script collision prediction horizon0.5s; simulator class default differs'),
            kp=('mgr/params.py',7,'K_P orbit gain0.05; not the nominal goal gain'),
            circle_radius=('mgr/params.py',8,'C_R temporary circle radius0.3m; simulator.py36-37 initial c_r/max_radius')),
        'chen_impc_ob_si_tracking': dict(
            K=('impc_ob/src/SET.py',27,'author discrete horizon8 with h0.15s at29; shared h0.05s, also include K24'),
            scale_lengths=('impc_ob/src/uav.py',134,'existing adapter metric mapping; false retains author lengths, true uniformly maps radius0.3 to common radius; not an author-named parameter'),
            epsilon_gain=('impc_ob/src/uav.py',134,'multiplier on author epsilon0.20 and dependent E/W at150/152; gain1 preserves initialization, adapter-named parameter')),
    }
    result={}
    for method,parameters in specs.items():
        result[method]={}
        for key,(relative,line,note) in parameters.items():
            path=VENDOR/relative
            result[method][key]=dict(path=str(path.relative_to(HERE)),line=line,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),note=note)
    return result


def construct():
    from execution_policy import active_methods
    from scipy.stats import qmc
    variants=historical_variants()
    profiles=read(BASE/'scene_types_oracle_coverage_v3/paper_after_four_column/PROFILES_FROZEN.json')['profiles']
    menus={}
    for method in active_methods():
        rows=[];seen={}
        def add(options,label):
            options=normalize(method,options);key=canonical(options)
            if key in seen:
                rows[seen[key]]['origins'].append(label)
                return
            seen[key]=len(rows)
            rows.append(dict(index=len(rows)+1,options=options,origins=[label]))
        add(DEFAULTS[method],'verified_author_values_with_declared_common_plant_mapping')
        for index,options in enumerate(variants[method]):
            add(options,'historical_v1_variant_'+str(index+1))
        for profile in profiles:
            if profile['method']==method:
                add(profile['options'],'historical_v3_profile_'+profile['id'])
        if method=='chen_impc_ob_si_tracking':
            add(dict(K=24),'author_prediction_duration_1.2s_at_common_dt0.05')
        for axis in AXES[method]:
            for multiplier in (.5,2.):
                options=dict(DEFAULTS[method]);options[axis]*=multiplier
                add(options,f'single_factor_{axis}_times{multiplier:g}')
        assert len(rows)<=16,'Required historical/structured settings exceed16'
        sampler=qmc.Sobol(d=len(AXES[method]),scramble=True,seed=2026102901)
        # Generate a power-of-two block; use sequential accepted distinct points.
        for index,point in enumerate(sampler.random_base2(10)):
            if len(rows)>=32:break
            options=dict(DEFAULTS[method])
            for axis,u in zip(AXES[method],point):
                value=DEFAULTS[method][axis]*(.5+1.5*float(u))
                options[axis]=int(round(value)) if axis=='K' else value
            add(options,'sobol_point_'+str(index+1))
        assert len(rows)==32 and rows[0]['options']==DEFAULTS[method]
        menus[method]=dict(initial=rows[:16],extension=rows[16:],axes=list(AXES[method]),
            box='continuous author-value multipliers[0.5,2]; integer K rounded. Required historical variants and author-duration K24 are retained even outside this box.',
            seed=2026102901, default_origin='Author constants/entry values with documented common-plant adaptations; not a reproduction of the complete author experiment.')
    from index_ours_history import pools
    ours={key:dict(initial=[dict(index=i,spec=spec)for i,spec in enumerate(values[:16],1)],
                   extension=[dict(index=i,spec=spec)for i,spec in enumerate(values[16:32],17)])
          for key,values in pools().items()}
    return dict(methods=menus,ours=ours,parameter_provenance=provenance(),
        original_author_reference=dict(options={m:DEFAULTS[m]for m in active_methods()},
            mgr_adapter='execution_revisions/v001/source/paper_methods/mgr_author_unicycle_tracking/adapter.py',
            description='Separate original1.1physical-safety-ratio row; not an extra tuning candidate. Default menu item1uses the aligned environment values.'),
        initial_count=16,extension_count=16,extension_condition='Exact Emax16-Emax8 >=3 percentage points in first-round scene-equal success; at most one expansion.',
        training_states_round1=[0,1],training_states_round2=list(range(2,8)),shortlist=4,
        rank=['scene_equal_success_desc','collision_count_asc','unhandled_episode_count_asc','all_queue_capped_seconds_asc','menu_index_asc'],
        training_only=True,no_solver_settings_in_menu=True,
        ours_saturation_interpretation='Apply the common saturation rule to the frozen old pools: initial prefix16, conditional extension prefix17–32, with no new candidate generation or reordering.',
        withdrawn_methods=read(HERE/'METHOD_EXECUTION_POLICY.json')['withdrawn_methods'])


def main():
    from addendum_gates import require_before_menus
    require_before_menus()
    assert (HERE/'PRIOR_TASK_DELIVERED.json').exists()
    assert (HERE/'PORT_GATES.json').exists() and (HERE/'ENV_REUSE_PHASE_COMPLETE.json').exists()
    path=HERE/'PAPER_MENUS_FROZEN.json'
    value=construct()
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists(): assert path.read_text()==text,'Frozen parameter menu changed'
    else:path.write_text(text)
    print('Paper menus frozen',hashlib.sha256(path.read_bytes()).hexdigest())


if __name__=='__main__':main()
