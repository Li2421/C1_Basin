from load import *
from scipy.stats import spearmanr
def ens(run):
 z=np.load(D+f'predictions/{run}/ENSEMBLE.npz');return z
for run in ['full_neighbour','full_original','add_E','old_neighbour']:
 z=ens(run);print('==',run)
 for v in VAL:
  p=z[v+'__probability']  # (3,S,49) over all val states (training+validation positions?)
  sc=p.mean((0,1));y=val_y(v).mean((0,1));nb=None
  top=np.argsort(-sc)[:3]
  rho=spearmanr(sc,y).correlation
  print(f"{v:28s} pick {KEYS[top[0]]:6s} pred top3 {[KEYS[i] for i in top]} true best {KEYS[int(y.argmax())]} {y.max():.2f} | true(pick) {y[top[0]]:.2f} | spearman {rho:.2f} | pred range {sc.max()-np.median(sc):.3f} true range {y.max()-np.median(y):.3f}")
