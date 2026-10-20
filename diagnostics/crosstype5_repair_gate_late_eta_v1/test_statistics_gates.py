"""Gate missing-label semantics and exact equal-count bootstrap regression checks."""
from common import *
from gate_selection import choose_parameters,score_grid
from coverage_statistics import probability_best,per_state

def main():
    checks={}
    packet=dict(names=np.asarray(['a','b']),types=np.asarray(['sourceA','sourceB']),best_mu=np.full((2,2),.8),best_sd=np.full((2,2),.1),zero_mu=np.full((2,2),.75),best_score=np.full((2,2),np.nan),safety_score=np.ones((2,2)),features=np.zeros((2,24)))
    packet['features'][:,5]=1
    picked=choose_parameters(packet)
    checks['unlabeled_repairs_never_scored_as_predictions']=score_grid(packet,np.ones((2,2),bool))['success'] is None
    checks['gate_chooses_labeled_safety_without_outer_labels']=picked['critic']['k']==2 and picked['critic']['delta']==.05 and picked['critic']['scored_fraction']==1
    checks['probe_conservative_tie_break']=picked['probe']['tau']==0 and picked['probe']['rho']==1
    packet['safety_score'][:]=np.nan;fallback=choose_parameters(packet)
    checks['all_missing_gate_explicit_gap']=fallback['critic'].get('gap') is not None and fallback['probe'].get('gap') is not None
    checks['hypergeometric_exact_probability']=np.isclose(probability_best(2,2,6),.6)
    lo=np.zeros((1,75));hi=lo.copy();hi[0,0]=1
    rng=np.random.default_rng(79);values=[];order=True
    for i in range(2000):
        v,_=per_state(lo,hi,np.asarray(['a']),np.asarray(['s']),draws=True,rng=rng);values.append(v[0,1,0]);order=order and np.all(v[:,0,:10]<=v[:,1,:10])
    checks['unknown_endpoints_share_subsets']=order and all(v>=0 for v in values)
    checks['bootstrap_recomputes_subsets_with_correct_probability']=np.std(values)>0 and abs(float(np.mean(values))-9/56)<.003
    result=dict(utc=utc(),passed=all(checks.values()),checks=checks,no_rollouts=True,no_training=True)
    write(HERE/'STATISTICS_GATE_TESTS.json',result);print(json.dumps(clean(result),indent=2));assert result['passed']
    seal('STATISTICS_GATE_TESTS.receipt.json',[HERE/'STATISTICS_GATE_TESTS.json',HERE/'test_statistics_gates.py',HERE/'gate_selection.py',HERE/'coverage_statistics.py'])
if __name__=='__main__':main()
