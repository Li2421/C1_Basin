"""Outcome map and 3-D axis-neighbor success graph for frozen Phase A."""
import json
from collections import Counter,defaultdict,deque
from itertools import product
from pathlib import Path
import numpy as np
from scipy.stats import beta
from diagnostics.success_basin_multimodality.setup import HERE,write

EVENTS=['success','deadlock','timeout','collision']

def one_sided(k,n,alpha=.05):
    return [float(beta.ppf(alpha,k,n-k+1)) if k else 0.,
            float(beta.ppf(1-alpha,k+1,n-k)) if k<n else 1.]

def two_sided(k,n,alpha=.05):
    return [float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.,
            float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.]

def key(eta):return tuple(float(x) for x in eta)

def main():
    p=json.loads((HERE/'protocol.json').read_text());rows=[]
    for s in p['primary_states']:
        m=json.loads((HERE/'raw'/f"phase_a_{s['state_id']}"/'manifest.json').read_text());rows+=m['records']
    groups=defaultdict(list)
    for r in rows:groups[(r['state_id'],key(r['eta']))].append(r)
    cells=[]
    for (sid,eta),rs in sorted(groups.items()):
        c=Counter(r['outcome'] for r in rs);n=len(rs);unknown=c[None];ks=c['success']
        lcb,ucb=one_sided(ks,n)
        if unknown or n<16:label='UNKNOWN_CELL'
        elif lcb>=p['classification']['q_success']:label='SUCCESS_CELL'
        elif ucb<=.20:label='FAILURE_CELL'
        else:label='UNKNOWN_CELL'
        cells.append({'state_id':sid,'eta':list(eta),'n':n,'counts':{x:c[x] for x in EVENTS},
          'solver_unknown':unknown,'probabilities':{x:c[x]/n for x in EVENTS},
          'exact_95_intervals':{x:two_sided(c[x],n) for x in EVENTS},
          'Q_S_one_sided_95':[lcb,ucb],'classification':label,
          'mean_steps':float(np.mean([r['steps'] for r in rs]))})
    write('success_map.json',{'criterion':p['classification'],'cells':cells,
      'total_rollouts':len(rows),'total_steps':sum(r['steps'] for r in rows),
      'outcome_counts':dict(Counter(r['outcome'] for r in rows))})
    axes=p['phase_a_design']['axes'];axis=[axes['goal'],axes['safe'],axes['relative']]
    allpoints=set(product(*axis))
    def neighbors(x):
        idx=[axis[i].index(x[i]) for i in range(3)]
        for d in range(3):
            for step in [-1,1]:
                q=idx.copy();q[d]+=step
                if 0<=q[d]<len(axis[d]):yield tuple(axis[i][q[i]] for i in range(3))
    result={}
    for state in p['primary_states']:
        sid=state['state_id'];by={key(c['eta']):c for c in cells if c['state_id']==sid};success={x for x,c in by.items() if c['classification']=='SUCCESS_CELL'}
        remain=set(success);comps=[]
        while remain:
            root=remain.pop();comp={root};todo=[root]
            while todo:
                for y in neighbors(todo.pop()):
                    if y in remain:remain.remove(y);comp.add(y);todo.append(y)
            comps.append(sorted(comp))
        comps.sort(key=len,reverse=True)
        unknown={x for x,c in by.items() if c['classification']=='UNKNOWN_CELL'}
        failure={x for x,c in by.items() if c['classification']=='FAILURE_CELL'}
        sep=[]
        for i in range(len(comps)):
            for j in range(i+1,len(comps)):
                a,b=min(((x,y) for x in comps[i] for y in comps[j]),key=lambda q:np.linalg.norm(np.asarray(q[0])-q[1]))
                sep.append({'components':[i,j],'minimum_distance':float(np.linalg.norm(np.asarray(a)-b)),
                            'closest_points':[list(a),list(b)]})
        result[sid]={'success_cell_count':len(success),'unknown_cell_count':len(unknown),'failure_cell_count':len(failure),
          'component_count':len(comps),'components':[[list(x) for x in c] for c in comps],
          'component_sizes':[len(x) for x in comps],'inter_component':sep,
          'known_success_eta_present':(1.,0.,.25) in success}
    sets=[{key(c['eta']) for c in cells if c['state_id']==s['state_id'] and c['classification']=='SUCCESS_CELL'} for s in p['primary_states']]
    common=sorted(set.intersection(*sets))
    write('connected_components.json',{'adjacency':'one lattice coordinate changes by one grid index','states':result,
      'common_success_cells_all_three_states':[list(x) for x in common],
      'finite_sampling_warning':'Components are empirical graph components, never a proof of mathematical disconnectedness.'})
    print(json.dumps({'outcomes':Counter(r['outcome'] for r in rows),'states':result,
                      'common_success_cells':len(common)},default=lambda x:dict(x),indent=2))

if __name__=='__main__':main()
