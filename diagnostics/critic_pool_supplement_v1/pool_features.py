"""Source-only normalization, exact-cell curves and empirical neighbour responses."""
from common import *
from design import *
import numpy as np
CENTER=np.array([1.,0.,.75]);RADIUS=np.array([1.,1.5,.75])
@lru_cache(None)
def pool():return {r['name']:r for r in read(HERE/'TRAINING_POOL.json')}
@lru_cache(None)
def rows(name):
 r=pool()[name];assert sha(r['rows_path'])==r['rows_sha256']
 with np.load(r['rows_path'],allow_pickle=False)as z:return {k:z[k]for k in z.files}
@lru_cache(None)
def response_tables(name):
 d=rows(name);out={}
 for k in [-1,0,1,2]:
  indices=np.arange(len(d['y']))if k==-1 else np.flatnonzero(d['ctl']==k)
  if not len(indices):continue
  groups={}
  for i in indices:
   key=etahex(d['eta'][i]);v=groups.setdefault(key,[d['eta'][i],0.,0.]);v[1]+=float(d['success'][i]);v[2]+=float(d['n'][i])
  values=[groups[k]for k in sorted(groups)];out[k]=(np.array([v[0]for v in values])-CENTER)/RADIUS,np.array([v[1]/v[2]for v in values]),np.array([v[2]for v in values])
 return out

def response(name,E):
 E=np.asarray(E,float);table=response_tables(name);values=[]
 for k in range(3):
  X,Y,N=table.get(k,table[-1]);w=N[None]*np.exp(-np.sum((E[:,None]-X[None])**2,-1)/(.08));values.append((w@Y)/np.maximum(w.sum(1),1e-12))
 return np.array(values)

def neighbour(names,c,eta,ctl,exclude=None):
 names=sorted(n for n in names if n!=exclude);allc=np.array([pool()[n]['c_prime']for n in sorted(set(names)|({exclude}if exclude is not None else set()))]);sigma=allc.std(0);sigma=np.where(sigma<1e-9,1,sigma)
 # The caller supplies the full source-pool sigma, so excluding a query does not refit distances.
 return neighbour_with_sigma(names,c,eta,ctl,sigma)

def neighbour_with_sigma(names,c,eta,ctl,sigma,exclude=None):
 names=sorted(n for n in names if n!=exclude);C=np.array([pool()[n]['c_prime']for n in names]);dist=np.sqrt(np.sum(((C-np.asarray(c))/sigma)**2,1));order=np.argsort(dist,kind='stable')[:3];near=[names[i]for i in order];E=(np.asarray(eta)-CENTER)/RADIUS
 unique={};inverse=[]
 for e in E:
  key=tuple(e)
  if key not in unique:unique[key]=len(unique)
  inverse.append(unique[key])
 U=np.array(list(unique));values=np.mean([response(n,U)for n in near],axis=0);inverse=np.asarray(inverse);g=values[np.asarray(ctl),inverse];ga=values.mean(0)[inverse]
 lg=lambda p:np.log(np.clip(p,.02,.98)/(1-np.clip(p,.02,.98)))
 return np.column_stack([lg(g),lg(ga),np.full(len(E),float(dist[order[0]]))]),dict(names=near,distances=dist[order].tolist())

@lru_cache(None)
def menu_curve(name):
 d=rows(name);lookup={etahex(a['eta_a']):i for i,a in enumerate(menu())};success=np.zeros(49);count=np.zeros(49,int);seen=[set()for _ in range(49)]
 for i,eta in enumerate(d['eta']):
  a=lookup.get(etahex(eta))
  if a is None:continue
  success[a]+=d['y'][i];count[a]+=1;seen[a].add((int(d['state'][i]),int(d['ctl'][i])))
 coverage=np.array([len(v)for v in seen])/(pool()[name]['state_count']*3);q=np.divide(success,count,out=np.full(49,np.nan),where=count>0);q[coverage<.5]=np.nan
 return q,coverage

def baseline(names,c,k=None):
 names=sorted(names);C=np.array([pool()[n]['c_prime']for n in names]);sigma=C.std(0);sigma=np.where(sigma<1e-9,1,sigma);dist=np.sqrt(np.sum(((C-c)/sigma)**2,1));near=[names[i]for i in np.argsort(dist,kind='stable')[:k]]if k else names
 curves=np.array([menu_curve(n)[0]for n in near]);finite=np.isfinite(curves);num=np.where(finite,curves,0).sum(0);den=finite.sum(0);q=np.divide(num,den,out=np.full(49,np.nan),where=den>0);assert np.isfinite(q).any();best=int(np.nanargmax(q))
 return dict(index=best,candidate=menu()[best],sources=near,source_count_per_candidate=den.tolist(),scores=[None if not np.isfinite(v)else float(v)for v in q])

def dataset(identifier,names):
 names=sorted(names);meta=HERE/'datasets'/f'{identifier}.json'
 if meta.exists():
  r=read(meta);assert sha(r['path'])==r['sha256'];return r
 C=np.array([pool()[n]['c_prime']for n in names]);cm=C.mean(0);cs=C.std(0);sigma=np.where(cs<1e-9,1,cs);cs=np.where(cs<1e-6,1,cs)
 H=np.concatenate([rows(n)['h'][rows(n)['state']]for n in names]);hm=H.mean(0);hs=H.std(0);hs=np.where(hs<1e-6,1,hs)
 raw=[];details={};ys=[];prob=[];scene_ids=[]
 for ni,n in enumerate(names):
  d=rows(n);g,detail=neighbour_with_sigma(names,pool()[n]['c_prime'],d['eta'],d['ctl'],sigma,n);details[n]=detail
  raw.append(np.column_stack([np.repeat(((C[ni]-cm)/cs)[None],len(d['y']),0),(d['h'][d['state']]-hm)/hs,np.eye(3)[d['ctl']],(d['eta']-CENTER)/RADIUS,g]));ys.append(d['y']);prob.extend([1/(len(names)*len(d['y']))]*len(d['y']));scene_ids.extend([ni]*len(d['y']))
 X=np.concatenate(raw);assert X.shape[1]==27 and np.isfinite(X).all()and all(n not in r['names']for n,r in details.items());gm=X[:,-3:].mean(0);gs=X[:,-3:].std(0);gs=np.where(gs<1e-6,1,gs);X[:,-3:]=(X[:,-3:]-gm)/gs;p=HERE/'datasets'/f'{identifier}.npz';p.parent.mkdir(exist_ok=True);np.savez_compressed(p,x=X.astype(np.float32),y=np.concatenate(ys).astype(np.float32),prob=np.asarray(prob)/np.sum(prob),scene=np.array(scene_ids))
 r=dict(name=identifier,names=names,path=str(p),sha256=sha(p),c_mean=cm.tolist(),c_scale=cs.tolist(),distance_sigma=sigma.tolist(),h_mean=hm.tolist(),h_scale=hs.tolist(),g_mean=gm.tolist(),g_scale=gs.tolist(),near=details,rows=len(X),input_order='c_prime12,h6,controller3,eta3,g3',normalization_source_only=True)
 write(meta,r);return r

def target_x(definition,data):
 names=definition['names'];h=np.asarray(data['h']);eta=np.asarray([a['eta_a']for a in menu()]);S=len(h);C=3;A=49
 H=np.broadcast_to(h[None,:,None],(C,S,A,6)).reshape(-1,6);ctl=np.broadcast_to(np.arange(3)[:,None,None],(C,S,A)).ravel();E=np.broadcast_to(eta[None,None],(C,S,A,3)).reshape(-1,3);g,detail=neighbour_with_sigma(names,data['c_prime'],E,ctl,np.asarray(definition['distance_sigma']))
 c=(np.asarray(data['c_prime'])-definition['c_mean'])/definition['c_scale'];X=np.column_stack([np.repeat(c[None],len(H),0),(H-definition['h_mean'])/definition['h_scale'],np.eye(3)[ctl],(E-CENTER)/RADIUS,(g-definition['g_mean'])/definition['g_scale']]);return X.astype(np.float32),(C,S,A),detail
