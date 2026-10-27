"""Eight LOSO folds; original v2 priors/normalization/internal tuning, full-only critic."""
from common import *
from design import *
from scenes import features
from descriptors_v2 import descriptor,ranges,target_guard
from anchored import prior
from models import select
import numpy as np

def normalization(inputs,sources,field):
 hh=np.vstack([np.asarray(inputs[n]['h'])[inputs[n]['training_state_positions']]for n in sources])
 cc=np.vstack([np.repeat(np.asarray(inputs[n][field])[None],len(inputs[n]['training_state_positions']),axis=0)for n in sources])
 hs=hh.std(0);cs=cc.std(0)
 return dict(h_mean=hh.mean(0).tolist(),h_scale=np.where(hs<1e-6,1,hs).tolist(),c_mean=cc.mean(0).tolist(),c_scale=np.where(cs<1e-6,1,cs).tolist(),fit_scenes=sources,fit_state_indices={n:[inputs[n]['state_indices'][i]for i in inputs[n]['training_state_positions']]for n in sources})

def main():
 assert(HERE/'HETEROGENEITY_CHECK.json').exists();tables=read(HERE/'LABEL_TABLES.json');scenes={s['name']:s for s in read(HERE/'SCENES.json')};evals=read(HERE/'EVALUATION_STATES.json')['scenes'];names=list(tables);inputs={};etas=np.asarray([a['eta_a']for a in menu()]);outer=[];inner={};tasks_in=[];tasks_out=[];guards={}
 for n in names:
  scene=scenes[n];d=tables[n];states=scene['states'];assert [s['index']for s in states]==d['state_indices']
  h=[features(dict(scene,states=[s]))[-6:]for s in states];cp=descriptor(scene).tolist()
  inputs[n]=dict(h=h,c=scene['features'][:12],c_raw=scene['features'][:12],c_prime=cp,state_indices=d['state_indices'],controllers=list(CONTROL_IDS),controller_types=list(CONTROL_TYPES),controller_one_hot=np.eye(8)[[TYPES.index(t)for t in CONTROL_TYPES]].tolist(),training_state_positions=[i for i,s in enumerate(states)if s['index']%4!=3],validation_state_positions=[i for i,s in enumerate(states)if s['index']%4==3],evaluation_state_positions=[d['state_indices'].index(i)for i in evals[n]])
 write(HERE/'INPUTS_V2.json',dict(scene_inputs=inputs,etas=etas.tolist(),eta_keys=ETA_KEYS,controller_order=TYPES,labels_not_in_input_features=True))
 write(HERE/'INPUTS.json',read(HERE/'INPUTS_V2.json'))
 def definition(name,sources,targets,kind,baseline=None):
  arrays={}
  for n in sources:
   d=inputs[n];arrays.update({n+'__h':np.asarray(d['h']),n+'__c':np.repeat(np.asarray(d['c_prime'])[None],len(d['h']),axis=0),n+'__onehot':np.asarray(d['controller_one_hot']),n+'__y':np.asarray(tables[n]['y'],float),n+'__state_indices':np.asarray(d['state_indices']),n+'__train':np.asarray(d['training_state_positions']),n+'__val':np.asarray(d['validation_state_positions'])})
  folder=HERE/'folds'/name;folder.mkdir(parents=True,exist_ok=True);label=folder/'source_labels.npz';np.savez_compressed(label,**arrays)
  out=dict(name=name,kind=kind,sources=sources,targets=targets,descriptor='c_prime',normalization=normalization(inputs,sources,'c_prime'),source_labels_path=str(label),training_data_sha256=sha(label),baselines=baseline or{},generator=kind=='loso',modes=['full'],ranges=ranges(inputs,sources))
  write(folder/'definition.json',out);write(folder/'PRIORS.json',prior(tables,sources,etas));return out
 for target in names:
  sources=[n for n in names if n!=target];f=np.asarray([scenes[n]['features']for n in sources]);q=np.asarray([np.asarray(tables[n]['y']).mean((0,1))for n in sources]);sel=select(f,q,scenes[target]['features'],etas)
  baseline={target:dict(select=sel,global_fixed=ETA_KEYS[sel['global_fixed']],knn=ETA_KEYS[sel['knn']])};name='loso__'+target;node=definition(name,sources,[target],'loso',baseline)
  assert read(HERE/'folds'/name/'PRIORS.json')['best_index']==sel['global_fixed'];links={'full':{}}
  for t in sources:
   subset=[n for n in sources if n!=t];key=uid('inner',dict(sources=subset,descriptor='c_prime',seed=17,kind='anchored_critic'))
   if key not in inner:
    inner[key]=definition(key,subset,[n for n in names if n not in subset],'inner');tasks_in.append(dict(fold=key,kind='critic',mode='full',seed=17,source_scene_count=len(subset)))
   links['full'][t]=dict(training_node=key,mode='full',target=t,guard=target_guard(inputs,t,inner[key]['ranges']))
  node['inner_links']=links;write(HERE/'folds'/name/'definition.json',node);outer.append(node);guards[name]={target:dict(scale_independent=target_guard(inputs,target,node['ranges']))}
  for mode in ['full','generator']:
   for seed in [17,23,41]:tasks_out.append(dict(fold=name,kind='generator'if mode=='generator'else'critic',mode=mode,seed=seed,source_scene_count=len(sources)))
 write(HERE/'FOLDS.json',outer);write(HERE/'INNER_NODES.json',list(inner.values()));write(HERE/'GUARD.json',dict(passed=True,folds=guards))
 write(HERE/'TRAIN_INNER_TASKS.json',tasks_in);write(HERE/'TRAIN_OUTER_TASKS.json',tasks_out);write(HERE/'TRAINING_PLAN.json',dict(outer_folds=len(outer),internal_fits=len(tasks_in),outer_fits=len(tasks_out),total_fits=len(tasks_in)+len(tasks_out),no_raw_or_no_prior_ablations=True,identical_internal_source_sets_deduplicated=True))
 seal('TRAINING_FROZEN.receipt.json',[HERE/n for n in ['INPUTS_V2.json','INPUTS.json','LABEL_TABLES.json','FOLDS.json','INNER_NODES.json','TRAIN_INNER_TASKS.json','TRAIN_OUTER_TASKS.json','GUARD.json','HETEROGENEITY_CHECK.json','train_v2.py','train_model.py','networks.py','anchored.py','predict_v2.py','descriptors_v2.py','models.py','prepare_learning.py']],all_source_only=True)
 print('Training plan:',len(tasks_in),'internal +',len(tasks_out),'outer fits',flush=True)
if __name__=='__main__':main()
