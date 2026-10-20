"""Source-set cached models; target label files are never opened here."""
from common import *
from selector_train40 import fit,Critic
from selector_features40 import apply_transform
from selector_data40 import MENU
from shared_rollout_db.src.rollout_db import uid
from flax import serialization
import jax,jax.numpy as jnp

def source_path(scene):
    if scene!='ring_exchange':return HERE/'selector_labels40'/f'{scene}.npz'
    pointer=read(HERE/'HISTORICAL_SELECTOR_SOURCE40_CURRENT.json');receipt=HERE/pointer['receipt'];assert sha(receipt)==pointer['receipt_sha256']
    path=HERE/pointer['dataset'];assert sha(path)==pointer['dataset_sha256'];return path

def load_sources(sources):
    assert 'G' not in sources and len(sources)==len(set(sources))
    paths=[source_path(s) for s in sorted(sources)]
    X=[];y=[];n=[];q=[];scenes=[];names=[];arms=[]
    for scene,path in zip(sorted(sources),paths):
        z=np.load(path);count=len(z['y']);X.append(z['X']);y.append(z['y']);n.append(z['n']);q.append(z['q']);scenes.extend([scene]*count)
        if len(z['controller_names'])==count:names.extend(z['controller_names'].tolist());arms.extend(z['arms'].tolist())
        else:names.extend(np.repeat(z['controller_names'],32*7).tolist());arms.extend(np.tile(MENU,count//7).tolist())
    assert len(names)==sum(map(len,y)) and len(arms)==len(names)
    return dict(X=np.concatenate(X).astype(np.float32),y=np.concatenate(y).astype(np.float32),n=np.concatenate(n).astype(np.float32),q=np.concatenate(q).astype(np.float32),scenes=np.asarray(scenes),names=np.asarray(names),arms=np.asarray(arms),source_paths=[str(p) for p in paths],source_hashes={str(p):sha(p) for p in paths})

def model_id(sources):
    return uid('sources',dict(sources=sorted(sources),files={s:sha(source_path(s)) for s in sorted(sources)},config_sha256=sha(HERE/'SELECTOR40_CONFIG_FROZEN.json'),training_source_sha256=sha(HERE/'selector_train40.py')))

def fit_sources(sources):
    cfg=read(HERE/'SELECTOR40_CONFIG_FROZEN.json');data=load_sources(sources);mid=model_id(sources);folder=HERE/'fits40'/mid;folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'MODEL_FROZEN.receipt.json'
    if receipt.exists():
        assert all(sha(p)==h for p,h in read(receipt)['files'].items());return mid
    assert len(sources)>=2,'Original leave-one-source-scene CV requires at least two source scenes'
    X,y,n,sc=data['X'],data['y'],data['n'],data['scenes'];hist=[]
    for held in sorted(sources):
        train=sc!=held;valid=sc==held;assert (n[train]>0).any() and (n[valid]>0).any()
        for seed in cfg['fit_seeds']:
            hist.append(fit(X,y,n,train,valid,seed,folder/'cv'/held/f'seed{seed}',cfg['steps']))
            print('SELECTOR_CV',mid,held,seed,flush=True)
    mean=np.asarray(hist).mean(0);assert np.isfinite(mean).all();selected=cfg['steps'][int(mean.argmin())]
    write(folder/'SELECTED_STEP.json',dict(utc=utc(),selected_step=selected,mean_validation_nll=mean,source_scenes=sorted(sources),source_paths=data['source_paths'],source_hashes=data['source_hashes'],fit_seed_ensemble=cfg['fit_seeds'],source_only_preprocessing=True))
    for seed in cfg['fit_seeds']:fit(X,y,n,np.ones(len(X),bool),np.zeros(len(X),bool),seed,folder/'full'/f'seed{seed}',[selected])
    paths=list(folder.rglob('*.msgpack'))+list(folder.rglob('transform.npz'))+[folder/'SELECTED_STEP.json',HERE/'selector_train40.py',HERE/'selector_model40.py',HERE/'SELECTOR40_CONFIG_FROZEN.receipt.json']
    seal(str(receipt.relative_to(HERE)),paths,source_scenes=sorted(sources),source_paths=data['source_paths'],source_G_rows=0,source_only_normalization_and_step=True)
    return mid

def source_mean(sources):
    data=load_sources(sources);sc=data['scenes'];cn=data['names'];arms=data['arms'];rates=[];upper_rates=[];coverage={}
    for arm in MENU:
        scene_rates=[];scene_upper=[];coverage[arm]={}
        for scene in sorted(sources):
            controllers=sorted(set(cn[(sc==scene)&(arms==arm)].tolist()));cr=[];cu=[]
            for name in controllers:
                ix=(sc==scene)&(arms==arm)&(cn==name);cr.append(float(np.mean(data['y'][ix]/data['q'][ix])));cu.append(float(np.mean((data['y'][ix]+data['q'][ix]-data['n'][ix])/data['q'][ix])))
            if cr:scene_rates.append(float(np.mean(cr)));scene_upper.append(float(np.mean(cu)))
            coverage[arm][scene]=dict(controllers=len(controllers),rows=int(((sc==scene)&(arms==arm)).sum()),available=bool(cr))
        rates.append(float(np.mean(scene_rates)) if scene_rates else -float('inf'))
        upper_rates.append(float(np.mean(scene_upper)) if scene_upper else -float('inf'))
    possible=[arm for arm,upper in zip(MENU,upper_rates) if upper>=max(rates)]
    return dict(chosen_arm=MENU[int(np.argmax(rates))],rates=rates,upper_rates=upper_rates,possible_chosen_arms_under_unknown_bounds=possible,coverage=coverage)

def predict(target,sources,tag):
    assert target not in sources and 'G' not in sources
    cfg=read(HERE/'SELECTOR40_CONFIG_FROZEN.json');mid=model_id(sources);folder=HERE/'fits40'/mid;receipt=folder/'MODEL_FROZEN.receipt.json';assert receipt.exists()
    assert all(sha(p)==h for p,h in read(receipt)['files'].items());z=np.load(HERE/'selector_inputs40'/f'{target}.npz');X=z['X'];selected=read(folder/'SELECTED_STEP.json')['selected_step'];net=Critic();base=net.init(jax.random.PRNGKey(17),jnp.zeros((1,33),jnp.float32));probs=[]
    for seed in cfg['fit_seeds']:
        full=folder/'full'/f'seed{seed}';tr=np.load(full/'transform.npz');params=serialization.from_bytes(base,(full/f'step{selected}.msgpack').read_bytes());tx=apply_transform(X.reshape(-1,33),tr);probs.append(np.asarray(jax.nn.sigmoid(net.apply(params,jnp.asarray(tx)))).reshape(X.shape[:-1]))
    probabilities=np.mean(probs,axis=0);choice=np.argmax(probabilities.mean(1),axis=-1)
    out=HERE/'selector_choices40'/f'{target}_{tag}.json';write(out,dict(utc=utc(),target_family=target,source_scenes=sorted(sources),source_model_id=mid,controller_names=z['controller_names'].tolist(),state_uids=z['state_uids'].tolist(),chosen_arms=[MENU[int(i)] for i in choice],probabilities=probabilities,source_mean=source_mean(sources),target_label_file_opened=False,G_source_rows=0))
    seal(str(out.with_suffix('.receipt.json').relative_to(HERE)),[out,receipt,HERE/'selector_inputs40'/f'{target}.npz',HERE/'selector_model40.py'],before_target_label_analysis=True,target_in_train=False,target_in_validation=False,target_in_preprocessing=False,target_label_file_opened=False)

if __name__=='__main__':fit_sources(read(HERE/sys.argv[1])[int(sys.argv[2])]['sources'])
