"""Frozen representation diagnostics; no physical rollout or model adaptation."""
import json,copy
from collections import defaultdict
import numpy as np
import jax,jax.numpy as jnp
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from . import representation as rep,build,train
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a

def proposals(row,models,seed_identity=None):
    gm,gp,cm,cp=models;sc=row['scenario'];x=rep.batch([rep.entities(build.scene(row))])
    raw=gm.apply(gp,x)[0]
    noise=jnp.asarray(np.random.default_rng(a.learn.stable_int('generator-v1-proposals',41,sc,seed_identity or row['state_uid'])).standard_normal((16,3)),jnp.float32)
    etas=np.asarray([a.learn.eta_mean(raw),*a.learn.eta_from_noise(raw[None].repeat(16,0),noise)],float)
    xx={k:np.repeat(v,17,axis=0) for k,v in x.items()}
    logits=np.asarray(cm.apply(cp,xx,jnp.asarray((etas-a.learn.CENTER)/a.learn.RADIUS)))
    return {'etas':etas.tolist(),'logits':logits.tolist(),'scores':(1/(1+np.exp(-logits))).tolist(),
            'selected_index':int(np.argmax(logits)),'proposal_hash':a.digest(etas.tolist())}

def run():
    rows=build.rows();models=train.load_models();gm,gp,cm,cp=models
    x=rep.batch([rep.entities(build.scene(r)) for r in rows]);enc=rep.Encoder()
    h=np.asarray(enc.apply({'params':gp['params']['physical_encoder']},x));hc=np.asarray(enc.apply({'params':cp['params']['physical_encoder']},x))
    np.save(build.OUT/'generator_embeddings.npy',h);np.save(build.OUT/'critic_embeddings.npy',hc)
    dd=np.linalg.norm(h[:,None]-h[None,:],axis=-1);np.fill_diagonal(dd,np.inf)
    aliases=np.argwhere(np.triu(dd<1e-7,1));assert not len(aliases),'Exact learned h alias needs inspection'
    labels=defaultdict(dict)
    for e in pq.read_table(build.DATA/'eta_labels.parquet').to_pylist():
        key=tuple(np.round(rep.decode(e['eta_raw']),8))
        labels[e['state_uid']][key]={'robust':e['robust_15of16'],
            'q':e['success_count']/e['seed_count'] if e['seed_count']>=16 and not e['numerical_failure_count'] else None}
    near=[];physical=[]
    for row in rows:
        e=rep.entities(build.scene(row));physical.append(np.r_[e['agents'].mean(0),e['agents'].max(0),
           e['pairs'].mean((0,1)),e['obstacles'].mean((0,1)),e['globals']])
    physical=np.asarray(physical)
    for i,row in enumerate(rows):
        j=int(np.argmin(dd[i]));u,v=labels[row['state_uid']],labels[rows[j]['state_uid']];common=set(u)&set(v)
        certified=[k for k in common if u[k]['robust'] is not None and v[k]['robust'] is not None]
        pos1={k for k in certified if u[k]['robust']};pos2={k for k in certified if v[k]['robust']}
        qkeys=[k for k in common if u[k]['q'] is not None and v[k]['q'] is not None]
        qa=np.asarray([u[k]['q'] for k in qkeys]);qb=np.asarray([v[k]['q'] for k in qkeys])
        correlation=float(spearmanr(qa,qb).statistic) if len(qa)>=3 and np.std(qa)>0 and np.std(qb)>0 else None
        near.append({'state_uid':row['state_uid'],'nearest_uid':rows[j]['state_uid'],'scenario':row['scenario'],
            'neighbor_scenario':rows[j]['scenario'],'h_distance':float(dd[i,j]),
            'physical_summary_distance':float(np.linalg.norm(physical[i]-physical[j])),
            'common_certified_eta':len(certified),'robust_jaccard_common_tested':len(pos1&pos2)/len(pos1|pos2) if pos1|pos2 else None,
            'common_full_Q_eta':len(qkeys),'Q_mean_absolute_difference':float(np.mean(np.abs(qa-qb))) if len(qa) else None,
            'Q_ranking_spearman':correlation})
    build.dump('aliasing.json',{'exact_alias_pairs':aliases.tolist(),'nearest_neighbors':near,
        'note':'Jaccard uses only commonly evaluated certified eta; unknown/missing evidence is not a negative.'})
    # Diagnostic nearest-centroid probe; bookkeeping is never fed to the model.
    scs=sorted({r['scenario'] for r in rows});centers=np.asarray([h[[i for i,r in enumerate(rows) if r['scenario']==sc and r['split']=='train']].mean(0) for sc in scs])
    val=[i for i,r in enumerate(rows) if r['split']=='validation'];guess=np.argmin(np.linalg.norm(h[val,None]-centers[None],axis=-1),axis=1)
    probe={'validation_accuracy':float(np.mean([scs[g]==rows[i]['scenario'] for i,g in zip(val,guess)])),
           'fit':'train-only nearest centroid','interpretation':'Physical geometry can identify a scenario; this is not bookkeeping leakage.',
           'explicit_scene_onehot':False,'bookkeeping_counterfactual_max_error':0.}
    for row in rows[:6]:
        changed={**row,'scenario':'arbitrary_logging_name','state_id':'not_a_feature'}
        assert build.scene(changed)==build.scene(row)
    build.dump('scenario_probe.json',probe)
    # Matched proposal noise and identical eta for score comparisons.
    symmetry=[]
    for sc in ('four_way_intersection','ring_exchange'):
        cohort=sorted([r for r in rows if r['scenario']==sc],key=lambda r:a.digest('unified-symmetry|'+r['state_uid']))[:12]
        for row in cohort:
            s=build.scene(row);p0=proposals(row,models);baseetas=np.asarray(p0['etas'])
            for k in (0,1,2,3):
                ss=rep.transform(s,k*np.pi/2,(.73,-1.11),np.roll(np.arange(4),1),np.arange(len(s['obstacles']))[::-1])
                rr={**row,'conditioning':json.dumps({'physical_entities':ss})};p1=proposals(rr,models)
                etaerr=float(np.max(np.abs(baseetas-np.asarray(p1['etas']))))
                scoreerr=float(np.max(np.abs(np.asarray(p0['scores'])-p1['scores'])))
                assert etaerr<2e-5 and scoreerr<2e-5,(sc,k,etaerr,scoreerr)
                symmetry.append({'scenario':sc,'state_uid':row['state_uid'],'rotation':90*k,
                    'proposal_max_error':etaerr,'score_max_error':scoreerr,'top1_same':p0['selected_index']==p1['selected_index'],
                    'passive_policy_chart_transform':True})
    build.dump('learned_metamorphic.json',{'tests':symmetry,'max_proposal_error':max(s['proposal_max_error'] for s in symmetry),
        'max_score_error':max(s['score_max_error'] for s in symmetry),'top1_consistency':float(np.mean([s['top1_same'] for s in symmetry]))})
    # Portable vector scatter: this environment does not include matplotlib.
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="330" viewBox="0 0 920 330"><rect width="920" height="330" fill="white"/>']
    for panel,key in enumerate(('physical_summary_distance','Q_mean_absolute_difference')):
        rr=[v for v in near if v[key] is not None];xmax=max(v['h_distance'] for v in rr);ymax=max(v[key] for v in rr)
        left=50+panel*450
        svg.append(f'<path d="M{left},35 V270 H{left+380}" fill="none" stroke="black"/>')
        svg.append(f'<text x="{left}" y="20" font-size="12">{key}</text><text x="{left+80}" y="300" font-size="12">Nearest h distance (0 to {xmax:.3g})</text>')
        svg.append(f'<text x="{left}" y="32" font-size="10">max y = {ymax:.3g}</text>')
        for v in rr:
            color=('#3975b7','#dd7b22','#379969')[scs.index(v['scenario'])]
            cx=left+380*v['h_distance']/max(xmax,1e-12);cy=270-230*v[key]/max(ymax,1e-12)
            svg.append(f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="2.3" fill="{color}" opacity="0.7"/>')
    svg.append('<text x="45" y="322" font-size="11">Blue: Double; orange: Four-Way; green: Ring. Exact values and IDs in aliasing.json.</text></svg>')
    (build.OUT/'nearest_neighbor_diagnostics.svg').write_text(''.join(svg))
    raw=np.asarray(gm.apply(gp,x));mu,sigma=a.learn.dist_params(jnp.asarray(raw))
    build.dump('generator_distribution.json',{'mean_sigma':float(np.mean(sigma)),'median_sigma':float(np.median(sigma)),
       'sigma_by_dimension':np.asarray(sigma).mean(0).tolist(),'location_boundary_fraction':float(np.mean(np.abs(np.tanh(np.asarray(mu)))>.99))})
    return {'exact_aliases':len(aliases),'scenario_probe':probe,'symmetry_tests':len(symmetry)}

if __name__=='__main__':print(json.dumps(run(),indent=2))
