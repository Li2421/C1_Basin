"""Only generated fixed eta values need execution; the 49 menu labels are reused."""
from common import *
from scenes import make_new
from replay import iter_array
import numpy as np,subprocess

OLD=ROOT/'diagnostics/cross_scene_correction_v1'

@lru_cache(None)
def scenes():return {s['name']:s for s in read_evidence(OLD/'SCENES.json')if s['family']!='extrapolation'}

@lru_cache(None)
def selected():return read_evidence(OLD/'SELECTED_CONTROLLERS.json')

@lru_cache(None)
def literature_templates():
    wanted={n for n in read(HERE/'SCENE_SET.json')['kept']if n.startswith('lit_')};out={}
    evidence(OLD/'main_CASES.json')
    for c in iter_array(OLD/'main_CASES.json'):
        if c['state']['scene']not in wanted:continue
        if c['requested_action']!='eta13':continue
        key=canonical([c['state']['scene'],c['controller_payload']['nominal']['controller'],c['state']['index']])
        assert key not in out;out[key]=c
    write(HERE/'LITERATURE_REQUEST_TEMPLATES.json',out);return out

def request(name,controller,index,eta):
    scene=scenes()[name];state=next(s for s in scene['states']if s['index']==index)
    ctl=next(c for c in selected()[name].values()if c is not None and c['controller']==controller)
    key='pipeline_fixed_'+uid('eta',etahex(eta)).split('_',1)[1]
    action=dict(key=key,eta_a=list(map(float,eta)))
    if scene['family']!='literature':c=make_new(scene,ctl,state,action,0)
    else:
        template=literature_templates()[canonical([name,controller,index])]
        payload={k:copy.deepcopy(v)for k,v in template['controller_payload'].items()if k not in ('method','eta_b','theta')}
        c=make_case(state,'flow',key,template['scenario_payload'],payload,0,eta,template['state_provenance'])
        c['rng_state_uid']=template.get('rng_state_uid',template['state_uid'])
    c['pipeline_scene']=name;return c

def verify_freezes():
    allf=read(HERE/'ALL_FOLDS_FROZEN.receipt.json');assert allf['folds']==31
    for p,h in allf['files'].items():
        assert sha(p)==h,p
        for source,digest in read(p)['files'].items():assert sha(source)==digest,source

def design(stage):
    verify_freezes();known={etahex(a['eta_a']):i for i,a in enumerate(actions())};cases={};refs=[]
    for fold in read(HERE/'FOLDS.json'):
        if not fold['generator']:continue
        if stage=='pilot'and fold['kind']!='loso':continue
        choices=read(HERE/'folds'/fold['name']/'CHOICES.json')
        for name,row in choices.items():
            if stage=='pilot':
                # Conservative fixed choice for timing, using an already qualified
                # conflict-search instance when available. No capability re-test.
                types=row['controller_types'];ci=types.index('conflict_based_search')if 'conflict_based_search'in types else 0
                work=[(ci,j,a,'pilot')for j in range(min(2,len(row['state_indices'])))for a in (49,50)]
            else:
                work=[]
                for ci in range(len(row['controllers'])):
                    for j in range(len(row['state_indices'])):
                        work.append((ci,j,49,'generator_mean'))
                        work.append((ci,j,row['pipeline'][ci][j],'pipeline'))
            for ci,j,a,method in work:
                eta=row['candidate_etas'][j][a];hex_=etahex(eta)
                reference=dict(fold=fold['name'],scene=name,controller=row['controllers'][ci],state_index=row['state_indices'][j],
                    method=method,chosen_index=a,eta=eta)
                if a<49 or hex_ in known:
                    reference.update(source='historical_49_label',historical_action=known[hex_]);refs.append(reference);continue
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
