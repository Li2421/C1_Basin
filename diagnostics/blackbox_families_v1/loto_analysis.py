from common import *
from expanded import old
from flax import serialization
import jax,jax.numpy as jnp
with np.load(HERE/'EXPANDED_TRAINING_DATA.npz') as z:X=z['X'];y=z['y'];n=z['n'];names=list(z['names'])
net=old.net_for('A3');result={}
for family in ['Ring Flow','Gaussian behavior cloning','Decentralized model predictive control']:
 folder=HERE/'expanded/loto'/family;config=read(folder/'17/FIT.json');indices=np.asarray(config['held_controllers']);xx=X[indices];pred=[]
 for seed in [17,23,41]:
  ck=folder/str(seed)/'model.msgpack';pa=serialization.from_bytes(net.init(jax.random.PRNGKey(0),jnp.zeros((1,29),jnp.float32)),ck.read_bytes());pred.append(np.asarray(jax.nn.sigmoid(net.apply(pa,jnp.asarray(xx.reshape(-1,29))))).reshape(xx.shape[:-1]))
 pp=np.mean(pred,axis=0).mean(1);choice=pp.argmax(-1);lo=y[indices];hi=lo+(1-n[indices]);ci=np.arange(len(indices));rate=lo.mean(1)[ci,choice];upper=hi.mean(1)[ci,choice];oracle=lo.mean(1).max(-1);oraclehi=hi.mean(1).max(-1);result[family]=dict(controller_names=[names[i] for i in indices],selected_arms=[['F10','F13','S500','RHR'][i] for i in choice],success_pct=[float(rate.mean()*100),float(upper.mean()*100)],in_sample_menu_regret_pp=[float((oracle-upper).mean()*100),float((oraclehi-rate).mean()*100)],evaluation='held-family source tuning states, DEVELOPMENT only',selected_step=read(HERE/'EXPANDED_STEP_SELECTION.json')['selected_step'])
write(HERE/'LEAVE_ONE_FAMILY_DEVELOPMENT.json',dict(utc=utc(),evidence_level='DEVELOPMENT',families=result))
