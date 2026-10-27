import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from analyze import load
from gap_ci import gap_once, ci_states
M=load(sys.argv[2:]); sub=sys.argv[1].split(',')
p=gap_once(M,sub,np.random.default_rng(0),reps=200); lo,hi=ci_states(M,sub)
print(f'{len(sub)} scenes (fixed, states resampled): one global eta below own by {100*p:+.1f} pp, 95% interval [{100*lo:+.1f}, {100*hi:+.1f}]')
