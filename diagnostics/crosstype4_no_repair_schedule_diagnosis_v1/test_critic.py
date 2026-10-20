"""Encoding, source isolation, architecture and oracle leakage checks."""
from common import *
from b_candidates import catalog,encode
from critic_model import initialize,metrics,source_transform,sampling_weights
from legacy_critic_model import Critic
from a_statistics import oracle_prediction,full_values,coverage_values
from a_uncertainty import difference
import jax,jax.numpy as jnp

def main():
    checks={};etas=np.arange(1,25,dtype=np.float64).reshape(8,3)/100;rows=catalog(etas);unique=[r for r in rows if r['unique_representative']];checks['441_unique_candidates']=len(unique)==441
    checks['zero_identity']=rows[483]['eta_a']==[0.,0.,0.] and rows[483]['eta_b']==[0.,0.,0.]
    checks['encoding13']=all(len(encode(a))==13 for a in unique);checks['progress_encoding']=all(encode(a)[9]==0 and encode(a)[12]==1 for a in unique if a['switch_mode']=='progress')
    checks['zero_stage_flags']=all(encode(a)[10]==float(a['early_index']==8) and encode(a)[11]==float(a['late_index']==8) for a in unique)
    new=initialize('full',17);old=Critic().init(jax.random.PRNGKey(17),jnp.zeros((1,100)),jnp.zeros((1,33)),jnp.zeros((1,24)),jnp.zeros((1,10)))
    a=new['params'];b=old['params'];checks['only_action_kernel_shape_changes']=a['action_encoder']['kernel'].shape==(13,32) and all(np.asarray(v).tobytes()==np.asarray(b[k][j]).tobytes() for k in a for j,v in a[k].items() if (k,j)!=('action_encoder','kernel'))
    data=dict(names=np.asarray(['a','b','held']),types=np.asarray(['A','B','T']),features=np.asarray([[1.]*24,[3.]*24,[999.]*24]),h=np.ones((2,100),np.float32),c=np.ones((2,33),np.float32),labels=np.ones((3,2,2,1),np.int8))
    first=source_transform(data,['A','B']);data['features'][2]=-12345;second=source_transform(data,['A','B']);checks['held_feature_isolation']=json.dumps(clean(first),sort_keys=True)==json.dumps(clean(second),sort_keys=True);w=sampling_weights(data,['A','B']).reshape(3,2,2);checks['held_gradient_zero']=np.all(w[2]==0) and np.isclose(w.sum(),1)
    # If one state is duplicated, both copies must be absent from training.
    # A leaked copy would choose pair0 on held state0 and yield success 1.
    labels=np.asarray([[1,0],[0,1],[0,1]],float);v,degenerate=oracle_prediction(labels,np.asarray([0,0,1]));checks['oracle_duplicate_holdout']=v==0 and degenerate==0
    _,degenerate=oracle_prediction(labels,np.asarray([0,0,0]));checks['oracle_single_state_no_label_fallback']=degenerate==3
    bounds=difference((np.asarray([0.]),np.asarray([1.]),np.asarray(['same'])),(np.asarray([0.]),np.asarray([1.]),np.asarray(['same'])));checks['same_UID_unknown_cancels']=np.array_equal(bounds,[0,0])
    # Opposite-seed selected time, not a label-dependent choice on the test seed.
    d={k:np.zeros((2,2)) for k in ['old_selected','selected_progress','human_progress',*[f'selected_{t}' for t in (200,300,400,500,600)],*[f'human_{t}' for t in (200,300,400,500,600)]]};d['selected_200'][:]=[1,0];d['selected_300'][:]=[0,1];checks['fixed_time_seed_crossfit']=full_values(d,np.arange(2))[2]==0
    indices=np.asarray([[[3,9,53,483]]]);labels=np.asarray([[[[1],[0],[1],[0]]]],np.int8);p=np.asarray([[[[.9,.1,.8,.2]]]]*3);m=metrics(p,labels,np.asarray(['A']),['A'],indices)['A'];checks['ranking_groups']=m['ranking']['zero_vs_nonzero']['identifiable_pairs']>0
    result=dict(utc=utc(),passed=all(checks.values()),checks=checks,action_dimension=13,old_architecture_except_input_unchanged=True);write(HERE/'CRITIC_TESTS.json',result);seal('CRITIC_TESTS.receipt.json',[HERE/n for n in ('CRITIC_TESTS.json','test_critic.py','critic_model.py','legacy_critic_model.py','b_candidates.py','a_uncertainty.py')]);assert result['passed'],result
if __name__=='__main__':main()
