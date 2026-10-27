import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from analyze import load, splits, own_honest, single_honest, cross_loss, KEYS, ETA
M=load(sys.argv[2:]); S=splits(M); sub=sys.argv[1].split(','); own={n:own_honest(M,S,n) for n in sub}
# which single eta wins globally (all states) and how each scene fares
F=np.mean([M[n].mean(0) for n in sub],0); g=int(np.argmax(F))
print('best single global eta (all states):',KEYS[g],np.round(ETA[KEYS[g]],2).tolist(),f'{100*F[g]:.1f}')
for n in sub:
    q=M[n].mean(0); b=int(np.argmax(q)); print(f"  {n:26s} own best {KEYS[b]:5s} {np.round(ETA[KEYS[b]],2).tolist()} own(honest) {100*own[n]:5.1f}  global eta here {100*q[g]:5.1f}  loss {100*(own[n]-q[g]):+5.1f}")
