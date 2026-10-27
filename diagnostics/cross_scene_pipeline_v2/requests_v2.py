"""Only frozen scene-level proposals; unchanged v1 corrected identity construction."""
from common import *
from requests_corrected import request,verify_freezes
import subprocess

def design(stage):
    verify_freezes();known={etahex(a['eta_a']):i for i,a in enumerate(actions())};cases={};refs=[]
    for fold in read(HERE/'FOLDS.json'):
        if not fold['generator']:continue
        for name,row in read(HERE/'folds'/fold['name']/'CHOICES.json').items():
            if stage=='pilot':
                if row['guard']['scene_out_of_range']:continue
                types=row['controller_types'];ci=types.index('conflict_based_search')if'conflict_based_search'in types else 0
                work=[(ci,s,a,'pilot')for s in range(min(2,len(row['state_indices'])))for a in(49,50)]
            else:
                work=[]
                for method in('v2','generator_only'):
                    choice=row['methods'][method]
                    if not choice['applicable']:continue
                    work.extend((c,s,a,method)for c,indices in enumerate(choice['indices'])for s,a in enumerate(indices))
            for ci,s,a,method in work:
                eta=row['candidate_etas'][a];hx=etahex(eta);ref=dict(fold=fold['name'],scene=name,controller=row['controllers'][ci],state_index=row['state_indices'][s],method=method,chosen_index=a,eta=eta)
                if a<49 or hx in known:
                    ref.update(source='historical_49_label',historical_action=known[hx]);refs.append(ref);continue
                c=request(name,row['controllers'][ci],row['state_indices'][s],eta)
                if c['rollout_uid']in cases:assert canonical(cases[c['rollout_uid']])==canonical(c)
                cases[c['rollout_uid']]=c;ref.update(source='generated_eta_rollout',rollout_uid=c['rollout_uid']);refs.append(ref)
    write(HERE/f'{stage}_REFERENCES.json',refs);return list(cases.values())

def prepare(stage):
    cases=design(stage)
    if stage=='pilot':
        main=design('main');union={c['rollout_uid']for c in cases+main}
        write(HERE/'NEW_ROLLOUT_BUDGET.json',dict(pilot_requests=len(cases),main_requests=len(main),unique_total=len(union),hard_upper_limit=1300,passed=len(union)<=1300))
        if len(union)>1300:
            write(HERE/'STOP.receipt.json',dict(utc=utc(),reason='NEW_ROLLOUT_BUDGET_EXCEEDED',count=len(union),limit=1300));raise RuntimeError('NEW_ROLLOUT_BUDGET_EXCEEDED')
    register(stage,cases)
    subprocess.run([sys.executable,'-B','-m','shared_rollout_db.plan','--manifest',str(HERE/f'{stage}_PLANNED_ROLLOUTS.json'),'--output',str(HERE/f'{stage}_OFFICIAL_PREFLIGHT.json')],cwd=ROOT,check=True)
    from queue_fast_v2 import build
    build(stage)

if __name__=='__main__':prepare(sys.argv[1])
