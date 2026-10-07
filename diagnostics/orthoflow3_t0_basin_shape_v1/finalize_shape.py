#!/usr/bin/env python3
"""Aggregate targeted Q64 probes and produce empirical morphology outputs; no rollouts."""
from __future__ import annotations
import csv,hashlib,json,math,statistics
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1');DIAG=ROOT/'diagnostics';HERE=DIAG/'orthoflow3_t0_basin_shape_v1';T0=DIAG/'orthoflow3_t0_basin_structure_v1';COMP=DIAG/'orthoflow3_t0_basin_completion_v1';MULTI=DIAG/'orthoflow3_t0_multiball_basin_learning_v1';MAN=DIAG/'orthoflow3_b63_manifold_representation_v1';BASIS=DIAG/'double_bottleneck_eta_basis_redesign/tools/bases.py';AFF=np.array([.875,0,.375]);SCALE=np.array([.75,1.,.75])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(e):return np.asarray(e,dtype=np.float64).tobytes().hex()
def nt(e):return (np.asarray(e,float)-AFF)/SCALE
def read(p):return list(csv.DictReader(open(p))) if Path(p).exists() else []
def write(p,rows,fields=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fields=fields or (list(dict.fromkeys(k for r in rows for k in r)) if rows else ['status'])
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def source_paths(sid):return [('t0_basin_structure',T0/'anchor_runs'/sid/'raw/pilot_rollouts.jsonl'),('t0_basin_completion',COMP/'raw'/sid/'raw/pilot_rollouts.jsonl'),('t0_multiball',MULTI/'stage_a'/sid/'raw/pilot_rollouts.jsonl'),('shape_targeted',HERE/'state_runs'/sid/'raw/pilot_rollouts.jsonl')]
def aggregate(states):
 out={};records=[]
 for st in states:
  sid=st['state_id'];ev={}
  for source,p in source_paths(sid):
   for line in p.read_text().splitlines():
    if not line.strip():continue
    r=json.loads(line)
    if r.get('state_id')!=sid or r.get('h_conditioning_identifier')!=st['h_conditioning_identifier'] or r.get('source_group')!=st['source_group']:continue
    e=np.asarray(r['eta'],dtype=np.float64);k=key(e);fi=int(r['future_index']);x=ev.setdefault(k,{'eta':e,'future':{},'sources':set(),'phases':set()});v=(bool(r['success']),str(r['outcome']))
    if fi in x['future'] and x['future'][fi]!=v:raise RuntimeError(('conflict',sid,k,fi))
    x['future'][fi]=v;x['sources'].add(source);x['phases'].add(str(r.get('phase','')))
  out[sid]=ev
  for k,x in ev.items():
   tr=len(x['future']);su=sum(v[0] for v in x['future'].values());cc=Counter(v[1] for v in x['future'].values());records.append({'state_id':sid,'eta_key_float64':k,'eta1':x['eta'][0],'eta2':x['eta'][1],'eta3':x['eta'][2],'successes':su,'trials':tr,'Q64_available':tr>=64,'Q64':su/64 if tr>=64 else '','B63':tr>=64 and su>=63,'deadlock':cc['safe_deadlock'],'timeout':cc['timeout'],'collision':cc['collision'],'numerical':cc['other_numerical'],'sources':';'.join(sorted(x['sources'])),'phases':';'.join(sorted(x['phases']))})
 return out,records
def svg_point(x,y,color):return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" fill-opacity=".78"/>'
def scale(values,lo,hi):return lo+(values-values.min())*(hi-lo)/max(values.max()-values.min(),1e-12)
def figure(sid,X,labels,frame,path):
 c=np.array([float(frame[f'center_t{i}']) for i in (1,2,3)]);U=np.array([[float(frame[f'u1_t{i}']) for i in (1,2,3)],[float(frame[f'u2_t{i}']) for i in (1,2,3)]]).T;n=np.array([float(frame[f'normal_t{i}']) for i in (1,2,3)]);Y=X-c;S=Y@U;N=Y@n
 x1=scale(S[:,0],50,550);y1=scale(S[:,1],440,60);x2=scale(S[:,0],650,1150);y2=scale(N,440,60);dots=[]
 for i,l in enumerate(labels):dots.append(svg_point(x1[i],y1[i],'#16803c' if l else '#b82222'));dots.append(svg_point(x2[i],y2[i],'#16803c' if l else '#b82222'))
 text=f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="500"><rect width="100%" height="100%" fill="white"/><text x="35" y="28" font-family="sans-serif" font-size="18">{sid}: observed Q64 only</text><text x="120" y="480" font-family="sans-serif">s1 vs s2</text><text x="730" y="480" font-family="sans-serif">s1 vs normal n</text><text x="35" y="55" font-family="sans-serif" font-size="12" fill="#16803c">green=B63</text><text x="130" y="55" font-family="sans-serif" font-size="12" fill="#b82222">red=Q64 non-B63</text><text x="310" y="55" font-family="sans-serif" font-size="12">unshown space=unknown</text><line x1="50" y1="450" x2="550" y2="450" stroke="black"/><line x1="50" y1="450" x2="50" y2="60" stroke="black"/><line x1="650" y1="450" x2="1150" y2="450" stroke="black"/><line x1="650" y1="450" x2="650" y2="60" stroke="black"/>{''.join(dots)}</svg>''';path.write_text(text)
def main():
 if sha(BASIS)!='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38':raise RuntimeError('basis hash')
 states=json.load(open(HERE/'fixed8_manifest.json'))['states'];ev,cached=aggregate(states);write(HERE/'cached_q64_manifest.csv',cached)
 initial=[];refine=[];results=[]
 for st in states:
  sid=st['state_id'];initial+=read(HERE/'state_runs'/sid/f'initial_results_{sid}.csv');refine+=read(HERE/'state_runs'/sid/f'refinement_results_{sid}.csv')
 results=initial+refine;write(HERE/'targeted_probe_results.csv',results)
 # Keep the required manifest as the complete frozen probe set while
 # preserving the initial role table separately for provenance.
 base_manifest=[r for r in read(HERE/'targeted_probe_manifest.csv') if r.get('category')!='SUCCESS_FAILURE_BOUNDARY_REFINEMENT']
 ref_manifest=read(HERE/'targeted_probe_refinement_manifest.csv')
 if ref_manifest: write(HERE/'targeted_probe_manifest.csv',base_manifest+ref_manifest)
 rmap={(r['state_id'],r['probe_id']):r for r in results};roles=read(HERE/'targeted_probe_roles.csv');frames={r['state_id']:r for r in read(MAN/'affine_plane_fit.csv')}
 interp=[];normal=[];boundary=[];holes=[];morph=[];cross=[];figure_paths=[]
 for st in states:
  sid=st['state_id'];frame=frames[sid];c=np.array([float(frame[f'center_t{i}']) for i in (1,2,3)]);U=np.array([[float(frame[f'u1_t{i}']) for i in (1,2,3)],[float(frame[f'u2_t{i}']) for i in (1,2,3)]]).T;nvec=np.array([float(frame[f'normal_t{i}']) for i in (1,2,3)])
  sr=[r for r in roles if r['state_id']==sid]
  # individual interpolation observations
  for r in sr:
   if r['category']!='SUCCESS_SUCCESS_INTERPOLATION':continue
   q=rmap[(sid,r['probe_id'])];interp.append({**r,'successes':q['successes'],'trials':q['trials'],'Q64':q['Q64'],'B63':q['B63'],'deadlock':q['deadlock'],'timeout':q['timeout'],'collision':q['collision']})
  # normal trials and contiguous thickness intervals
  nr=[r for r in sr if r['category']=='NORMAL_DIRECTION']
  for r in nr:
   q=rmap[(sid,r['probe_id'])];normal.append({**r,'successes':q['successes'],'trials':q['trials'],'Q64':q['Q64'],'B63':q['B63'],'record_type':'probe'})
  for anchor in sorted({str(r['anchor_rank']) for r in nr}):
   for sign in (-1,1):
    rr=sorted([r for r in nr if str(r['anchor_rank'])==anchor and int(float(r['normal_sign']))==sign],key=lambda r:float(r['normal_offset']));last=0.;first_fail=None
    for r in rr:
     if rmap[(sid,r['probe_id'])]['B63']=='True' and first_fail is None:last=float(r['normal_offset'])
     elif first_fail is None:first_fail=float(r['normal_offset'])
    normal.append({'state_id':sid,'record_type':'thickness_summary','anchor_rank':anchor,'normal_sign':sign,'robust_lower_bound':last,'failure_upper_bound':first_fail if first_fail is not None else 'NOT_OBSERVED','transition_interval':f'[{last},{first_fail}]' if first_fail is not None else f'>={last}'})
  # initial boundary plus at most one refinement
  for r in sr:
   if r['category'] not in ('SUCCESS_FAILURE_BOUNDARY','SUCCESS_FAILURE_BOUNDARY_REFINEMENT'):continue
   q=rmap[(sid,r['probe_id'])];boundary.append({**r,'successes':q['successes'],'trials':q['trials'],'Q64':q['Q64'],'B63':q['B63'],'deadlock':q['deadlock'],'timeout':q['timeout'],'collision':q['collision']})
  for r in sr:
   if r['category']!='HOLE_INTERIOR':continue
   q=rmap[(sid,r['probe_id'])];holes.append({**r,'successes':q['successes'],'trials':q['trials'],'Q64':q['Q64'],'B63':q['B63'],'deadlock':q['deadlock'],'timeout':q['timeout'],'collision':q['collision']})
  ir=[x for x in interp if x['state_id']==sid];hr=[x for x in holes if x['state_id']==sid];bycat=defaultdict(list)
  for x in ir:bycat[x['pair_category']].append(x)
  rate=lambda rr:sum(x['B63']=='True' for x in rr)/len(rr) if rr else math.nan
  fill=rate(ir);local=rate(bycat['local_1']+bycat['local_2']);medium=rate(bycat['medium']);long=rate(bycat['long_cross_sector']);hole_non=(sum(x['B63']!='True' for x in ir)/len(ir) if ir else math.nan);explicit_hole_fail=(sum(x['B63']!='True' for x in hr)/len(hr) if hr else math.nan)
  summ=[x for x in normal if x['state_id']==sid and x.get('record_type')=='thickness_summary'];pos=[float(x['robust_lower_bound']) for x in summ if int(x['normal_sign'])==1];neg=[float(x['robust_lower_bound']) for x in summ if int(x['normal_sign'])==-1];thick=(statistics.median(pos+neg) if pos+neg else 0.)
  q64=[r for r in cached if r['state_id']==sid and bool(r['Q64_available'])];B=[r for r in q64 if bool(r['B63'])];X=np.array([nt([float(r['eta1']),float(r['eta2']),float(r['eta3'])]) for r in B]);Y=X-c;S=Y@U;N=Y@nvec;span1=float(S[:,0].max()-S[:,0].min());span2=float(S[:,1].max()-S[:,1].min());spann=float(N.max()-N.min());anis=max(span1,span2)/max(spann,1e-12)
  if (not math.isnan(hole_non) and hole_non>=.30) or (not math.isnan(explicit_hole_fail) and explicit_hole_fail>=.50):cls='HOLED_OR_INTERLEAVED_REGION'
  elif not math.isnan(local) and local>=.75 and not math.isnan(long) and long<=.50:cls='MULTICOMPONENT_LIKE'
  elif not math.isnan(fill) and fill>=.80 and thick>=.10:cls='THICK_ANISOTROPIC_VOLUME'
  elif not math.isnan(fill) and fill>=.80 and thick<.10:cls='THIN_FILLED_SLAB'
  elif not math.isnan(local) and local>=.75 and (math.isnan(long) or long<.80):cls='LOBED_CONNECTED_REGION'
  else:cls='UNDERRESOLVED'
  bnd=[x for x in boundary if x['state_id']==sid];refcount=sum(x['category'].endswith('REFINEMENT') for x in bnd);coherent=sum(1 for x in bnd if x['category']=='SUCCESS_FAILURE_BOUNDARY')
  morph.append({'state_id':sid,'interpolation_probes':len(ir),'interpolation_B63_rate':fill,'local_B63_rate':local,'medium_B63_rate':medium,'long_B63_rate':long,'midpoint_hole_rate':hole_non,'explicit_hole_probes':len(hr),'explicit_hole_failure_rate':explicit_hole_fail,'normal_thickness_lower_median':thick,'normal_positive_lower_median':statistics.median(pos) if pos else math.nan,'normal_negative_lower_median':statistics.median(neg) if neg else math.nan,'tangent1_span':span1,'tangent2_span':span2,'normal_span':spann,'anisotropy_tangent_over_normal':anis,'boundary_initial_pairs':coherent,'boundary_refinements':refcount,'morphology':cls})
  cross.append({'state_id':sid,'morphology':cls,'aligned_tangent1_span':1.,'aligned_tangent2_span':span2/max(span1,1e-12),'aligned_normal_span':spann/max(span1,span2,1e-12),'interpolation_fill':fill,'normal_thickness_lower':thick,'anisotropy':anis,'B63_observations':len(B),'nonB63_observations':len(q64)-len(B)})
  fig=HERE/'figures'/f'{sid}_observed_q64.svg';figure(sid,np.array([nt([float(r['eta1']),float(r['eta2']),float(r['eta3'])]) for r in q64]),[bool(r['B63']) for r in q64],frame,fig);figure_paths.append(str(fig))
 write(HERE/'interpolation_results.csv',interp);write(HERE/'normal_thickness.csv',normal);write(HERE/'success_failure_boundary.csv',boundary);write(HERE/'hole_probe_results.csv',holes);write(HERE/'per_state_morphology.csv',morph);write(HERE/'cross_state_shape_comparison.csv',cross)
 classes=Counter(x['morphology'] for x in morph);maxclass,maxcount=classes.most_common(1)[0];ratios=[float(x['aligned_normal_span']) for x in cross];fills=[float(x['interpolation_fill']) for x in cross if not math.isnan(float(x['interpolation_fill']))]
 # A canonical shape requires near-uniform aligned extents as well as a
 # common morphology.  Six of eight is not enough when the remaining
 # tangential/normal scale spread is material.
 if maxcount>=7 and statistics.pstdev(ratios)<=.10:crosscls='CANONICAL_SHAPE_APPROXIMATELY_SHARED'
 elif maxcount>=5:crosscls='SHARED_MORPHOLOGY_STATE_DEPENDENT_DEFORMATION'
 elif sum(c in ('HOLED_OR_INTERLEAVED_REGION','MULTICOMPONENT_LIKE') for c in classes)>=2 and sum(c in ('THIN_FILLED_SLAB','THICK_ANISOTROPIC_VOLUME','LOBED_CONNECTED_REGION') for c in classes)>=2:crosscls='BASIN_TOPOLOGY_STATE_DEPENDENT'
 elif len(classes)>=3:crosscls='MULTIPLE_BASIN_MORPHOLOGIES'
 else:crosscls='CROSS_STATE_SHAPE_UNDERRESOLVED'
 if crosscls=='SHARED_MORPHOLOGY_STATE_DEPENDENT_DEFORMATION' and maxclass in ('THIN_FILLED_SLAB','THICK_ANISOTROPIC_VOLUME'):rec='state-conditioned anisotropic slab/tube, contingent on future false-inclusion validation'
 elif maxclass=='LOBED_CONNECTED_REGION':rec='state-conditioned lobed implicit set'
 elif maxclass in ('HOLED_OR_INTERLEAVED_REGION','MULTICOMPONENT_LIKE') or crosscls in ('BASIN_TOPOLOGY_STATE_DEPENDENT','MULTIPLE_BASIN_MORPHOLOGIES'):rec='generic implicit feasibility field or explicitly separated conservative components'
 else:rec='no learning representation justified until targeted geometry validation is expanded'
 recommendation={'cross_state_classification':crosscls,'morphology_counts':dict(classes),'best_supported_representation':rec,'empirical_only':True,'unknown':'No finite observed grid establishes connectedness, interpolation safety, absence of holes, or a certified tube false-inclusion rate.'}
 dump(HERE/'representation_recommendation.json',recommendation)
 theory='''# Theory interpretation\n\nDefine the population basin as `B(h)={eta:Q(h,eta)>=1-epsilon}` under fixed t0 conditioning and persistent eta. Exact equality of `B(h)` across distinct h is not theoretically expected. If closed-loop outcome probability is smooth in h and eta and a boundary is regular, nearby states can induce smooth deformations of a shared boundary family. Topology may nevertheless change near coordination-critical events, safety-projection active-set changes, or deadlock bifurcations. Thus the scientifically appropriate hypothesis is a shared representation family with state-conditioned geometry—not one universal fixed shape. None of these smoothness or regularity assumptions is proven here; the observed Q64 samples provide only empirical evidence.\n''';(HERE/'theory_interpretation.md').write_text(theory)
 runt=[]
 for st in states:
  sid=st['state_id']
  for stage in ('initial','refinement'):
   p=HERE/'state_runs'/sid/f'{stage}_runtime.json'
   if p.exists():runt.append(json.load(open(p)))
 # Per-stage runtime counters can be cumulative after a resumed state run.
 # Count physical work directly from the new raw continuation records.
 raw_records=[]
 for st in states:
  p=HERE/'state_runs'/st['state_id']/'raw'/'pilot_rollouts.jsonl'
  if p.exists(): raw_records += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
 runtime={'new_continuations':len(raw_records),'physical_steps':sum(int(x.get('continuation_steps',0)) for x in raw_records),'max_state_wall_seconds':max((float(x['wall_seconds']) for x in runt),default=0),'max_gpu_shards':6,'stages':runt}
 dump(HERE/'runtime_statistics.json',runtime)
 final={'classification':crosscls,'morphology_counts':dict(classes),'representation_recommendation':rec,'cached_q64_eta_records':sum(1 for x in cached if bool(x['Q64_available'])),'new_targeted_eta':len({(x['state_id'],x['eta_key_float64']) for x in results}),'new_rollouts':runtime['new_continuations'],'critical_limitation':recommendation['unknown'],'next_experiment':'Frozen targeted validation of interpolation, normal offsets, and candidate gaps around the empirically favored representation; no learning until false-inclusion rate is measured.'}
 dump(HERE/'final_decision.json',final)
 table='\n'.join(f"| {x['state_id']} | {x['interpolation_B63_rate']:.2f} | {x['midpoint_hole_rate']:.2f} | {x['normal_thickness_lower_median']:.3f} | {x['anisotropy_tangent_over_normal']:.2f} | {x['morphology']} |" for x in morph)
 report=f'''# OrthoFlow3 t0 basin shape and cross-state morphology\n\n## Scope\n\nCached exact Q64 labels were aggregated before the frozen targeted probe batch. No model was trained. Every genuinely new probe used 64 matched future seeds.\n\n| State | interpolation B63 | interpolation non-B63 | normal lower bound | tangent/normal anisotropy | morphology |\n|---|---:|---:|---:|---:|---|\n{table}\n\n## Cross-state conclusion\n\n**{crosscls}**. Morphology counts: `{dict(classes)}`. The recommendation is **{rec}**. Alignment removes raw center/orientation differences but does not erase observed variation in tangential filling, normal thickness, or holes/boundary behavior.\n\n## Theory\n\nObserved basins are empirical samples of `B(h)`, not proven sets. Exact equality across h is not expected; a shared representation family with state-conditioned deformation is more plausible under smoothness/regularity assumptions, while topology can change near coordination or projection-active-set events. No theorem is claimed.\n\n## Limitation\n\nObserved labels establish only the sampled points. Unobserved space is unknown; this experiment cannot establish connectedness, interpolation safety outside tested trajectories, absence of holes, or a certified false-inclusion rate.\n\n## Next step\n\n{final['next_experiment']}\n''';(HERE/'final_report.md').write_text(report)
 required=['protocol.md','cached_q64_manifest.csv','targeted_probe_manifest.csv','targeted_probe_results.csv','interpolation_results.csv','normal_thickness.csv','success_failure_boundary.csv','hole_probe_results.csv','per_state_morphology.csv','cross_state_shape_comparison.csv','theory_interpretation.md','representation_recommendation.json','final_decision.json','runtime_statistics.json','final_report.md']
 dump(HERE/'manifest.json',{'experiment':'ORTHOFLOW3_T0_BASIN_SHAPE_AND_CROSS_STATE_MORPHOLOGY_V1','created_utc':datetime.now(timezone.utc).isoformat(),'orthoflow3_sha256':sha(BASIS),'new_rollouts':runtime['new_continuations'],'gpu_jobs_submitted':2,'networks_trained':0,'artifacts':{x:sha(HERE/x) for x in required},'figures':figure_paths})
 print(json.dumps(final,indent=2))
if __name__=='__main__':main()
