"""Type weighting, paired bootstrap identity and tied-rank checks."""
from common import *
from b_statistics import bootstrap,average_ranks
from a_uncertainty import full_bounds,coverage_bounds
from a_statistics import full_values,coverage_values

def main():
    labels=np.ones((3,4,8,2),np.int8);labels[2]=0;types=np.asarray(['A','A','B']);x,y=bootstrap(labels,types,np.ones(3,bool),np.random.default_rng(1),50);checks={'type_equal_not_controller_pooled':np.all(x[0]==.5),'paired_identical_methods':np.all(x[:,0]-x[:,1]==0),'known_upper_equals_lower':np.array_equal(x,y),'average_tie_ranks':np.array_equal(average_ranks([2,1,2,4]),[2.5,1,2.5,4])}
    # Bounds must contain the lower-label point and exactly cancel same UID.
    methods=['old_selected','selected_progress','human_progress',*[f'selected_{t}' for t in (200,300,400,500,600)],*[f'human_{t}' for t in (200,300,400,500,600)]];rng=np.random.default_rng(19);lower={k:rng.integers(2,size=(8,2)).astype(float) for k in methods};lower['proposal']=rng.integers(2,size=(8,56)).astype(float);upper={k:np.maximum(v,rng.integers(2,size=v.shape)) for k,v in lower.items()};uids={k:np.asarray([f'{k}-{j}' for j in range(v.size)]).reshape(v.shape) for k,v in lower.items()};d={0:lower,1:upper,'uids':uids};draw=np.asarray([0,0,1,2,3,4,5,7]);f=full_values(lower,draw);b=full_bounds(d,draw);c,_=coverage_values(lower,draw);bc=coverage_bounds(d,draw);checks['full_point_within_unknown_bounds']=np.all(f>=b[0]-1e-12) and np.all(f<=b[1]+1e-12);checks['coverage_point_within_unknown_bounds']=np.all(c>=bc[0]-1e-12) and np.all(c<=bc[1]+1e-12);checks['coverage_telescopes']=np.isclose(c[0],c[1:4].sum(),atol=1e-12)
    result=dict(utc=utc(),passed=all(checks.values()),checks=checks);write(HERE/'STATISTICS_TESTS.json',result);seal('STATISTICS_TESTS.receipt.json',[HERE/n for n in ('STATISTICS_TESTS.json','test_statistics.py','a_uncertainty.py','b_statistics.py')]);assert result['passed'],result
if __name__=='__main__':main()
