"""Publication table from already sealed rates and immutable labeled records."""
from core import *
from audit_utils import source_inventory
import csv,collections
MENU=['F10','F13','S300','S500']
def main():
    r=read(HERE/'results.json');names=r['eligible_controllers'];choices=read(HERE/'FROZEN_CHOICES.json')['controllers'];tc=read(HERE/'DEVELOPMENT_TEACHER_CHOICES.json')['controllers'];records={};out=[]
    for phase in ['main','teacher']:
        for f in (HERE/phase/'raw').glob('*.json'):
            x=read(f);c=x['case'];records[(phase,c['base_controller_name'],c['state_index'],c['seed'],c['arm'])]=x
    old=D/'decentralized_pipeline_v1';historic={}
    for c in read(old/'main_CASES.json'):
        if c['pipeline'] in ['local','central','shared','rhr'] and c['base_controller_name'] in names:
            x=read(old/'main/raw'/f"{c['rollout_uid']}.json");historic[(c['pipeline'],c['base_controller_name'],c['state_index'],c['seed'],c['arm'])]=x
    def add(family,policy,stats,select,getrow,note=''):
        rows=[getrow(n,s,seed,select(n)) for n in names for s in range(32) for seed in [0,1]];success=sum(x['success'] if not x['numerical_failure'] else 0 for x in rows)/len(rows);assert abs(success-stats['success']['mean'])<1e-12,(family,policy,success,stats['success']['mean'])
        out.append(dict(family=family,policy=policy,n=len(rows),success_lower=success,success_upper=stats.get('success_upper',success),CI95_low=stats['success']['CI95'][0],CI95_high=stats['success']['CI95'][1],collision_rate=sum(x['collision'] for x in rows)/len(rows),timeout_rate=sum(x['timeout'] for x in rows)/len(rows),unknown_rate=sum(x['numerical_failure'] for x in rows)/len(rows),note=note))
    student=lambda n,s,k,a:records[('main',n,s,k,a)]
    for arm,stats in r['student_table'].items():
        select=(lambda n:max(MENU,key=lambda a:r['controller_rates'][n][a])) if arm=='oracle' else (lambda n,a=arm:a);add('Student D-local',arm,stats,select,student,'oracle is posthoc menu reference' if arm=='oracle' else '')
    for key in ['A3_full','A0_full','G_E','S500','F10','A3_fixed','A0_fixed']:add('Student D-local',key,r['selectors'][key],lambda n,k=key:choices[n][k],student,'pre-evaluation frozen choice')
    for mode,table in r['teacher_tables'].items():
        for policy,stats in table.items():
            if policy in ['nominal only','safety baseline']:
                add('Teacher '+mode,policy,stats,lambda n,a=policy:a,lambda n,s,k,a:records[('teacher',n,s,k,a)]);continue
            if policy.startswith('B-RHR'):
                add('Teacher shared',policy,stats,lambda n:'B-RHR',lambda n,s,k,a:historic[('rhr',n,s,k,a)],'historical radius infinity; not D-local');continue
            if policy.endswith('_historical_R5'):sel=lambda n,m=policy[:2]:tc[n][m]
            elif policy=='always-S500':sel=lambda n:'S500'
            elif policy=='oracle':
                def sel(n,m=mode):
                    rates={a:sum(historic[(m,n,s,k,a)]['success'] if not historic[(m,n,s,k,a)]['numerical_failure'] else 0 for s in range(32) for k in [0,1])/64 for a in MENU};return max(MENU,key=lambda a:rates[a])
            else:sel=lambda n,a=policy:a
            add('Teacher '+mode,policy,stats,sel,lambda n,s,k,a,m=mode:historic[(m,n,s,k,a)],'DEVELOPMENT; historical R5 descriptor' if policy.endswith('_historical_R5') else 'DEVELOPMENT historical labels')
    with (HERE/'paper_table_complete.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    lines=['| Family | Policy | Success lower–upper | 95% CI | Collision | Timeout | Unknown |','|---|---|---:|---:|---:|---:|---:|']
    for x in out:lines.append(f"| {x['family']} | {x['policy']} | {100*x['success_lower']:.2f}–{100*x['success_upper']:.2f}% | [{100*x['CI95_low']:.2f}, {100*x['CI95_high']:.2f}]% | {100*x['collision_rate']:.2f}% | {100*x['timeout_rate']:.2f}% | {100*x['unknown_rate']:.2f}% |")
    lines+=['','Each row n=896. All success CIs are the sealed5,000 paired controller/state-cluster bootstrap results; this script only independently supplies observed collision/timeout/unknown rates. Oracle diagnostic rates use lower-label per-controller argmax; its upper success bound independently maximizes upper labels.','','Student A3/A0/G_E choices were frozen before evaluation. Teacher A3/A0 use historical R5 and frozen models, DEVELOPMENT only. Teacher D-local B-RHR was not requested/evaluated; the shown teacher RHR is D-shared(radius infinity). Teacher central/shared zero-correction safety baselines were not run and are not substituted with D-local.']
    (HERE/'paper_table_complete.md').write_text('\n'.join(lines)+'\n');source_inventory('imports_table_complete.json');print('Complete table rows',len(out))
if __name__=='__main__':main()
