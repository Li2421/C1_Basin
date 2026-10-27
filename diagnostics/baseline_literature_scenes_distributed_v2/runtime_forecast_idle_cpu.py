"""Estimate remaining time from landed evidence only; no new experiments."""
from pathlib import Path
import sys,os,json,collections,time
p=Path(__file__).resolve().parent;os.sched_setaffinity(0,{json.loads((p/'LOCAL_CPU_GUARD_STATUS.json').read_text())['local_cpu']});sys.path.insert(0,str(p))
from experiment import *
from atomic_write import write
started=time.monotonic();groups=collections.defaultdict(list);newgroups=collections.defaultdict(list);counts=collections.Counter();stage_counts=collections.Counter()
for stage in ['learning_labels_lit_circle_five']+[f'distributed_{kind}_{scene}' for kind in ('baseline','fixed') for scene in NAMES]:
    paths=list((HERE/stage/'raw').glob('*.json'))
    # All circle evidence; a deterministic representative sample elsewhere.
    if 'circle' not in stage:paths=sorted(paths)[::8]
    for path in paths:
        r=read(path);c=r['case'];nom=c['controller_payload']['nominal'];ctl=nom['controller'];scene=c['state']['scene'];seconds=r.get('elapsed_seconds',0)
        if stage.startswith('learning_labels'):
            counts[ctl]+=1
            if r.get('pure_geometry_cache_equivalence_receipt_sha256'):newgroups[ctl].append(seconds)
        if seconds>0 and not r.get('deterministic_source'):
            groups[(scene,ctl,'labels' if stage.startswith('learning_labels') else 'baseline' if '_baseline_' in stage else 'fixed')].append(seconds)
        stage_counts[stage]+=1
summary=[]
for (scene,ctl,kind),values in sorted(groups.items()):
    values.sort();summary.append(dict(scene=scene,controller=ctl,scope=kind,n=len(values),mean_seconds=sum(values)/len(values),median_seconds=sorted(values)[len(values)//2],p90_seconds=values[int(.9*(len(values)-1))],max_seconds=max(values)))
# Inspect manifest fields rather than inventing missing controllers.
write(HERE/'RUNTIME_FORECAST_EVIDENCE_IDLE_CPU.json',dict(utc=utc(),elapsed_analysis_seconds=time.monotonic()-started,summary=summary,label_counts=dict(counts),new_cache_label_statistics={k:dict(n=len(v),mean_seconds=sum(v)/len(v),median_seconds=sorted(v)[len(v)//2],max_seconds=max(v)) for k,v in newgroups.items()},sample_rule='All circle raws; sorted every eighth raw elsewhere; positive nonalias timings only',new_rollouts=0,new_training=0))
print(json.dumps(dict(label_counts=dict(counts),new_cache_label_statistics={k:dict(n=len(v),mean_seconds=sum(v)/len(v),median_seconds=sorted(v)[len(v)//2],max_seconds=max(v)) for k,v in newgroups.items()},circle_fixed=[x for x in summary if x['scene']=='lit_circle_five']),ensure_ascii=False,indent=2))
