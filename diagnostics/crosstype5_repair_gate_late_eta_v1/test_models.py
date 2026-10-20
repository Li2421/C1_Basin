"""Small meaningful model/protocol checks; no rollout or research fitting."""
from common import *
from candidate_menu import menu,choose,physical_key,METHODS
from critic_model import initialize,metrics,source_transform,sampling_weights
from schedule_generator import density_data,source_weights,log_prob,initialize as init_generator,ScheduleGenerator
import jax,jax.numpy as jnp

def main():
    checks={}
    frozen=np.asarray([[.82+.01*i,.1,.1] for i in range(8)])
    generated=np.stack([np.stack([frozen[i%8],frozen[(i+1)%8]]) for i in range(9)])
    rows=menu(frozen,generated);checks['dedup_no_partial_zero']=len({physical_key(r['action']) for r in rows})==len(rows) and all(not r['action']['eta_a']==[0.,0.,0.] for r in rows[:-1])
    requery=[r for r in rows if '切换时重查' in r['origins']];checks['requery_mean_early_not_diagonal']=len(requery)==16 and requery[0]['action']['late_source']=='requery'
    p=np.full((3,len(rows)),.5);p[:,-1]=.9
    selections,_=choose(rows,p,1,.02,.9,.1,np.asarray([0,0,0,0,0,1]))
    checks['high_safety_gates_choose_zero']=all(selections[m]==len(rows)-1 for m in METHODS[:6])
    params=initialize('full',17);checks['critic_only_action_kernel_expanded']=params['params']['action_encoder']['kernel'].shape==(14,32)
    n=3;action=np.zeros((n,2,40,14),np.float32);action[...,:6]=np.tile([.8,.1,.1],2)
    action[0,0,0,:3]=0
    labels=np.zeros((n,2,40,2),np.int8);labels[...,1]=-2;labels[:,0,1,0]=1;labels[1,1,2,0]=1
    data=dict(names=np.asarray(['a','b','c']),types=np.asarray(['source','source','outer']),features=np.arange(72,dtype=float).reshape(3,24),h=np.zeros((2,100),np.float32),c=np.zeros((2,33),np.float32),action=action,labels=labels)
    eta,pos,neg,base=density_data(data);w,gap=source_weights(data,['source'],pos,base)
    checks['generator_density_excludes_partial_zero']=not pos[0,0,0] and not neg[0,0,0]
    checks['generator_sampling_excludes_outer']=np.all(w[2]==0) and np.isclose(w.sum(),1)
    cw=sampling_weights(data,['source']).reshape(3,2,40);checks['critic_sampling_excludes_outer']=np.all(cw[2]==0)
    first=source_transform(data,['source']);data['features'][2]=1e12;second=source_transform(data,['source'])
    checks['preprocessing_excludes_outer']=all(np.array_equal(first[k],second[k]) for k in ('feature_fill','feature_mean','feature_scale'))
    gp=init_generator();raw=ScheduleGenerator().apply(gp,jnp.zeros((2,100)),jnp.zeros((2,33)),jnp.zeros((2,24)))
    grad=jax.grad(lambda rr:-log_prob(rr,jnp.asarray(eta[0,0,:2])).mean())(raw)
    checks['six_dim_density_gradient_finite']=raw.shape==(2,12) and np.isfinite(grad).all()
    # Sparse requery with same early/placeholder must stay non-diagonal in ranking.
    pred=np.asarray([[[[.8,.2,.9]]]]*3);lab=np.asarray([[[[1,-2],[0,-2],[0,-2]]]])
    packed=np.asarray([[[1,32,4]]]);r=metrics(pred,lab,np.asarray(['source']),['source'],packed)['source']['ranking']
    checks['sparse_rank_metadata']=r['cross']['identifiable_pairs']==2 and r['same_pair_different_time']['identifiable_pairs']==0
    result=dict(utc=utc(),checks=checks,passed=all(checks.values()),no_real_training=True,no_rollouts=True)
    write(HERE/'MODEL_TESTS.json',result)
    if result['passed'] and not (HERE/'MODEL_TESTS.receipt.json').exists():seal('MODEL_TESTS.receipt.json',[HERE/n for n in ('MODEL_TESTS.json','test_models.py','critic_model.py','schedule_generator.py','candidate_menu.py')])
    print(json.dumps(clean(result),indent=2));assert result['passed']
if __name__=='__main__':main()
