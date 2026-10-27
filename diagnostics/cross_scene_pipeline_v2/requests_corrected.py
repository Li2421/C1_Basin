"""Per-scene original-request construction; alter only eta-bound identities/metadata."""
from common import *
from replay import iter_array
from scenes import make_new
import subprocess

OLD=ROOT/'diagnostics/cross_scene_correction_v1'
LITERATURE=('lit_narrow_passage','lit_crowded_random','lit_circle_five','lit_swap8_obstacles')
BASE_CANDIDATES={'lit_circle_five':'sE05'}

@lru_cache(None)
def scenes():return {s['name']:s for s in read_evidence(OLD/'SCENES.json')if s['family']!='extrapolation'}

@lru_cache(None)
def selected():return read_evidence(OLD/'SELECTED_CONTROLLERS.json')

@lru_cache(None)
def original_requests():
    path=OLD/'main_CASES.json';digest=evidence(path);out={}
    for c in iter_array(path):
        if c['requested_action']!=BASE_CANDIDATES.get(c['state']['scene'],'eta13'):continue
        key=canonical([c['state']['scene'],c['controller_payload']['nominal']['controller'],c['state']['index']])
        assert key not in out;out[key]=c
    write(HERE/'ORIGINAL_REQUEST_INDEX.receipt.json',dict(source=str(path),source_sha256=digest,requests=len(out),
        base_candidate_by_scene=dict(default='eta13',**BASE_CANDIDATES),per_scene_state_controller=True,no_shared_scene_template=True))
    return out

def literature_templates():
    return {k:c for k,c in original_requests().items()if c['state']['scene']in LITERATURE}

def execution_entry(c):
    if c.get('cross_scene_new'):return 'runtime_new.run'
    if c['state']['scene']=='lit_swap8_obstacles':
        assert c.get('scene_revision')=='static_straight_corridors_20261023'
        return 'swap_runtime_user_fix_20261023.run'
    return 'runtime_correction_pa_memo_query_disk_candidate_spatial_mask_20261022.run'

def validate_scene_profile(c):
    name=c['state']['scene'];s=c['scenario_payload']
    if name=='lit_swap8_obstacles':
        assert c['version']=='swap8_si_user_fix_20261023'
        assert s['version']=='swap8_single_integrator_user_fix_20261023'
        assert c['scene_revision']==s['user_scene_revision']=='static_straight_corridors_20261023'
        for field in ('moving_policy','geometry_sha256','walls'):assert field in s
    elif name=='lit_circle_five':
        assert c['version']=='flow'and s['version']=='flow'
        assert execution_entry(c)=='runtime_correction_pa_memo_query_disk_candidate_spatial_mask_20261022.run'
    elif name in ('lit_narrow_passage','lit_crowded_random'):
        assert c['version']=='flow'and s['version']=='flow'
    else:
        assert c['cross_scene_new']and c['version']=='cross_scene_si_v1'
        scene=scenes()[name];ctl=c['controller_payload']['nominal']
        # Check the original make_new scenario contract independently. Keep the
        # historical scientific execution profile when running its certified
        # isolated copy, so relocating a cache does not alter request semantics.
        fresh=make_new(scene,ctl,c['state'],dict(key=c['requested_action'],eta_a=c['eta_a']),c['seed'])
        assert fresh['scenario_payload']==s and fresh['state_uid']==c['state_uid']
    assert c['theta']is None and c['controller_payload']['theta']is None

def from_original(original,eta):
    """A true identity reconstruction at the original eta, not a copied UID."""
    validate_scene_profile(original)
    c=copy.deepcopy(original);eta=list(map(float,eta));changed=etahex(eta)!=etahex(original['eta_a'])
    if changed:
        key='pipeline_fixed_'+uid('eta',etahex(eta)).split('_',1)[1]
        c['method']=key;c['requested_action']=key;c['controller_payload']['method']=key
        if 'canonical_execution_method'in c:c['canonical_execution_method']=key
        # These fields refer to the old eta's menu slot or completed result.
        # They cannot be asserted as provenance for a newly generated eta.
        for field in ('menu_index','reuse_source','reuse_source_sha256','alias_source'):c.pop(field,None)
    c['eta_a']=eta;c['eta_b']=list(eta);c['controller_payload']['eta_b']=list(eta)
    c['scenario_uid']=uid('scenario',c['scenario_payload'])
    c['state_uid']=uid('state',dict(scenario=c['scenario_uid'],physical={k:c['state'][k]for k in ('positions','velocities','goals')}))
    c['controller_uid']=uid('ctl',dict(scenario=c['scenario_uid'],payload=c['controller_payload']))
    c['eta_uid']=dbm.eta_identity(eta)[0];c['seed_key']=canonical(dict(future_index=c['seed']))
    c['rollout_uid']=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))
    assert c['scenario_uid']==original['scenario_uid']and c['state_uid']==original['state_uid']
    eta_fields={'method','requested_action','canonical_execution_method','menu_index','reuse_source','reuse_source_sha256','alias_source',
        'eta_a','eta_b','eta_uid','controller_uid','rollout_uid','controller_payload'}
    assert {k:v for k,v in c.items()if k not in eta_fields}=={k:v for k,v in original.items()if k not in eta_fields}
    assert {k:v for k,v in c['controller_payload'].items()if k not in ('eta_b','method')}=={k:v for k,v in original['controller_payload'].items()if k not in ('eta_b','method')}
    return c

def request(name,controller,index,eta):
    base=original_requests()[canonical([name,controller,index])]
    if name=='lit_circle_five':
        assert 'best_correction_search_v1/circle_'in base.get('reuse_source','')
    return from_original(base,eta)

def verify_freezes():
    allf=read(HERE/'ALL_FOLDS_FROZEN.receipt.json');assert allf['folds']==31
    for p,h in allf['files'].items():
        assert sha(p)==h,p
        for source,digest in read(p)['files'].items():assert sha(source)==digest,source
    check=read(HERE/'REQUEST_RECONSTRUCTION_CHECK.json');assert check['passed']and check['cases']>=10
    assert check['constructor_sha256']==sha(HERE/'requests_corrected.py')

def design(stage):
    verify_freezes();known={etahex(a['eta_a']):i for i,a in enumerate(actions())};cases={};refs=[]
    for fold in read(HERE/'FOLDS.json'):
        if not fold['generator']:continue
        if stage=='pilot'and fold['kind']!='loso':continue
        for name,row in read(HERE/'folds'/fold['name']/'CHOICES.json').items():
            if stage=='pilot':
                types=row['controller_types'];ci=types.index('conflict_based_search')if 'conflict_based_search'in types else 0
                work=[(ci,j,a,'pilot')for j in range(min(2,len(row['state_indices'])))for a in (49,50)]
            else:
                work=[(ci,j,a,m)for ci in range(len(row['controllers']))for j in range(len(row['state_indices']))
                    for a,m in ((49,'generator_mean'),(row['pipeline'][ci][j],'pipeline'))]
            for ci,j,a,method in work:
                eta=row['candidate_etas'][j][a];hx=etahex(eta)
                reference=dict(fold=fold['name'],scene=name,controller=row['controllers'][ci],state_index=row['state_indices'][j],method=method,chosen_index=a,eta=eta)
                if a<49 or hx in known:
                    reference.update(source='historical_49_label',historical_action=known[hx]);refs.append(reference);continue
                c=request(name,row['controllers'][ci],row['state_indices'][j],eta)
                if c['rollout_uid']in cases:assert canonical(cases[c['rollout_uid']])==canonical(c)
                cases[c['rollout_uid']]=c;reference.update(source='generated_eta_rollout',rollout_uid=c['rollout_uid']);refs.append(reference)
    write(HERE/f'{stage}_REFERENCES.json',refs)
    return list(cases.values())

def prepare(stage):
    cases=design(stage)
    if stage=='main':
        pilot=read(HERE/'pilot_CASES.json');union={c['rollout_uid']for c in cases+pilot}
        write(HERE/'NEW_ROLLOUT_BUDGET.json',dict(main_requests=len(cases),pilot_requests=len(pilot),unique_total=len(union),prespecified_approximate_upper_limit=2700))
        assert len(union)<=2700,('NEW_ROLLOUT_BUDGET',len(union))
    register(stage,cases)
    subprocess.run([sys.executable,'-B','-m','shared_rollout_db.plan','--manifest',str(HERE/f'{stage}_PLANNED_ROLLOUTS.json'),
        '--output',str(HERE/f'{stage}_OFFICIAL_PREFLIGHT.json')],cwd=ROOT,check=True)
    from queueing import build
    build(stage)

if __name__=='__main__':prepare(sys.argv[1])
