"""B15 counts on uniform-box Sobol points plus existing frozen candidate panel."""
import collections
import json
import sqlite3
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')


def main():
    protocol=json.loads((HERE/'robust_protocol.json').read_text())
    rows=[]
    for st in protocol['states']:
        uid=st['uid']
        for i in range(32):
            entry=dict(uid=uid,index=i,counts={})
            for ch in ('TT','FF'):
                base=json.loads((HERE/'panel_results'/f'{uid}_sobol64_{i:03d}_{ch}.json').read_text())
                r=[base]+[json.loads((HERE/'robust_results'/f'{uid}_{ch}_s{i:02d}_seed{s:02d}.json').read_text()) for s in range(1,16)]
                if any(x['success'] is None for x in r):raise RuntimeError((uid,i,ch,'unlabeled'))
                if any((x['rho']>0)!=x['success'] for x in r if x['rho'] is not None):raise RuntimeError((uid,i,ch,'rho sign'))
                count=sum(x['success'] for x in r)
                numerical=sum(x['termination']=='numerical_failure' for x in r)
                entry['counts'][ch]=dict(success=count,B15=count>=15,seed0=base['success'],
                    numerical_failures=numerical,B15_if_all_numeric_repaired=count+numerical>=15)
            a,b=entry['counts']['TT']['B15'],entry['counts']['FF']['B15']
            entry['classification']='both' if a and b else 'TT_only' if a else 'FF_only' if b else 'neither'
            rows.append(entry)
    states=[]
    for st in protocol['states']:
        ss=[r for r in rows if r['uid']==st['uid']]
        c=collections.Counter(r['classification'] for r in ss)
        states.append(dict(uid=st['uid'],points=len(ss),TT_B15=c['TT_only']+c['both'],FF_B15=c['FF_only']+c['both'],
                           robust_gain=c['FF_only'],robust_loss=c['TT_only'],counts=dict(c),
                           numerical_failures={ch:sum(r['counts'][ch]['numerical_failures'] for r in ss) for ch in ('TT','FF')},
                           potentially_unresolved_B15={ch:sum(r['counts'][ch]['B15_if_all_numeric_repaired'] and not r['counts'][ch]['B15'] for r in ss)
                                                       for ch in ('TT','FF')}))
    con=sqlite3.connect(HARD/'rollout.sqlite')
    frozen=json.loads((HARD/'confirmation_manifest.json').read_text())['states']
    old_by_scene={}
    for scene in ('ring_exchange','toy_give_way'):
        frozen_states=sorted((s for s in frozen if s['scenario']==scene),key=lambda s:s['uid'])
        old=[]
        for st in frozen_states:
            counts=collections.defaultdict(lambda:collections.defaultdict(list))
            for ch,i,text in con.execute('SELECT chain,eta_index,result_json FROM rollout WHERE state_uid=? AND chain IN ("TT","FF")',(st['uid'],)):
                counts[ch][i].append(json.loads(text)['success'])
            cls=collections.Counter()
            for i in range(16):
                a=sum(counts['TT'][i])>=15;b=sum(counts['FF'][i])>=15
                cls['both' if a and b else 'TT_only' if a else 'FF_only' if b else 'neither']+=1
            old.append(dict(uid=st['uid'],counts=dict(cls),TT_B15=cls['TT_only']+cls['both'],
                            FF_B15=cls['FF_only']+cls['both']))
        old_by_scene[scene]=old
    con.close()
    result=dict(sobol32_states=states,sobol32_points=rows,
                old_Q16_candidate_distribution=old_by_scene,
                warning='Q16 historical candidates are not uniform in the numerical box; their counts are not Lebesgue volume.')
    (HERE/'robust_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(5,4))
    x=np.arange(len(states));ax.bar(x-.15,[s['robust_gain'] for s in states],width=.3,label='FF only');
    ax.bar(x+.15,[s['robust_loss'] for s in states],width=.3,label='TT only')
    ax.set_xticks(x,[s['uid'].split('_')[-1] for s in states]);ax.set_ylabel('B15 point count / 32');ax.legend();fig.tight_layout()
    (HERE/'figures').mkdir(exist_ok=True);fig.savefig(HERE/'figures/robust_gain_loss.pdf');fig.savefig(HERE/'figures/robust_gain_loss.png',dpi=180);plt.close(fig)
    print(json.dumps(states,indent=2))


if __name__=='__main__':main()
