"""Original thirteen raw coordinates plus explicit requery indicator."""
from common import *
def encode(a):
    early=np.asarray(a['eta_a'],np.float64);late=np.asarray(a['eta_b'],np.float64)
    return [*early,*late,*(late-early),0. if a['switch_mode']=='progress' else a['switch_step']/700,float(np.all(early==0)),float(np.all(late==0)),float(a['switch_mode']=='progress'),float(a.get('late_source','given')=='requery')]
