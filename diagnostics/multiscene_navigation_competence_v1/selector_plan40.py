"""Fixed source combinations; identical source sets share one fitted model."""
from common import *
from selector_model40 import model_id,fit_sources,predict

def prepare():
    cfg=read(HERE/'SELECTOR40_CONFIG_FROZEN.json');gates=read(HERE/'AF_MULTIAGENT_CAPABILITY_GATES.json');families=gates['eligible_families'];historical=['ring_exchange'] if (HERE/'HISTORICAL_SELECTOR_SOURCE40_CURRENT.json').exists() else []
    models={};predictions=[];gaps=[]
    for target in families:
        requests=[('leave_one_family',[f for f in families if f!=target],False)]
        for m,combos in cfg['combinations'][target].items():
            for i,sources in enumerate(combos):requests.append((f'm{m}_combination{i}',sources,True))
        for tag,new_sources,strict in requests:
            missing=[f for f in new_sources if not (HERE/'selector_labels40'/f'{f}.npz').exists()]
            sources=historical+[f for f in new_sources if f not in missing]
            assert target not in sources and 'G' not in sources
            if (strict and missing) or len(sources)<2:
                gaps.append(dict(target=target,tag=tag,reason='Fixed combination contains blocked source families' if missing and strict else 'At least two source scenes required for unchanged leave-one-source-scene validation',missing_sources=missing,available_sources=sources));continue
            mid=model_id(sources);models.setdefault(mid,dict(model_id=mid,sources=sorted(sources)))
            predictions.append(dict(target=target,tag=tag,model_id=mid,sources=sorted(sources),partial_source_coverage=bool(missing or len(historical)<3),requested_training_family_count=len(new_sources),actual_training_family_count=len([s for s in sources if s in 'ABCDEF'])))
    p=HERE/'SELECTOR40_MODEL_PLAN_FROZEN.json';write(p,dict(utc=utc(),models=list(models.values()),predictions=predictions,gaps=gaps,historical_scenes=historical,target_families=families,model_cache='Exact source set plus immutable dataset/config/training hashes; no duplicate source-set training',source_G_rows=0))
    seal('SELECTOR40_MODEL_PLAN_FROZEN.receipt.json',[p,HERE/'selector_plan40.py',HERE/'selector_model40.py',HERE/'SELECTOR40_CONFIG_FROZEN.receipt.json'],before_target_label_analysis=True,target_excluded_from_all_fits=True)
    return p

def worker(index,count):
    import jax
    assert all(d.platform=='gpu' for d in jax.devices());jax.config.update('jax_enable_x64',False)
    plan=read(HERE/'SELECTOR40_MODEL_PLAN_FROZEN.json')
    for task in plan['models'][index::count]:
        assert model_id(task['sources'])==task['model_id'];fit_sources(task['sources'])

def freeze_predictions():
    plan=read(HERE/'SELECTOR40_MODEL_PLAN_FROZEN.json')
    for entry in plan['predictions']:
        out=HERE/'selector_choices40'/f"{entry['target']}_{entry['tag']}.receipt.json"
        if out.exists():assert all(sha(p)==h for p,h in read(out)['files'].items());continue
        predict(entry['target'],entry['sources'],entry['tag'])
    for gap in plan['gaps']:
        out=HERE/'selector_choices40'/f"{gap['target']}_{gap['tag']}.json"
        if not out.exists():write(out,dict(utc=utc(),status='GAP',**gap))
    paths=list((HERE/'selector_choices40').glob('*.receipt.json'))+[HERE/'SELECTOR40_MODEL_PLAN_FROZEN.receipt.json']
    seal('ALL_AF_CHOICES40_FROZEN.receipt.json',paths,before_any_target_evaluation_analysis=True,G_labels_used=False)

if __name__=='__main__':worker(int(sys.argv[2]),int(sys.argv[3])) if sys.argv[1]=='worker' else globals()[sys.argv[1]]()
