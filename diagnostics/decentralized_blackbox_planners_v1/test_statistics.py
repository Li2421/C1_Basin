from common import *
import stats as st
from metrics import features
G=guard()
def main():
 x=np.ones((4,32,2));z=np.zeros_like(x);tests={}
 tests['identical_paired_zero']=st.paired_success(x,z,x,z)['CI_unknown_envelope']==[0.,0.]
 tests['certain_success']=st.outcome_rate(x,z)['CI_unknown_envelope']==[1.,1.]
 tests['all_unknown_paired']=st.paired_success(z,x,z,x)['CI_unknown_envelope']==[-1.,1.]
 tests['undefined_conditional']=st.mean(np.full_like(x,np.nan))['estimate'] is None
 p=np.broadcast_to(np.array([[2.,0.],[0.,2.],[-2.,0.],[0.,-2.]]),(701,4,2)).copy();m=features(p,np.zeros((700,4,2)),np.zeros((4,2)));tests['stationary_140_robot_seconds']=m['stall_robot_seconds']==140 and m['mode']==0 and m['last_progress_seconds']==0 and m['moving_no_progress_robot_seconds']==0
 tests['copied_mechanism_logic_unchanged']=all((HERE/f).read_text().replace('from mechanism_core import *','from core import *')==(D/'teacher_student_line_closeout_v1'/f).read_text() for f in ['metrics.py','stats.py'])
 write(HERE/'STATISTICS_TESTS.json',dict(passed=all(tests.values()),checks=tests,guard=guard()));assert all(tests.values()),tests
if __name__=='__main__':main()
