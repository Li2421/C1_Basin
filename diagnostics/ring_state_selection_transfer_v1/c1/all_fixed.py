"""Check C1 oracle gain against the strongest full-panel fixed candidate, retaining K16."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from diagnostics.ring_state_selection_transfer_v1.study import read,write
H=Path(__file__).resolve().parent;r=read(H/'RESULTS.json');y=np.load(H/'labels.npz')['Y'];lo=(y==1).astype(float);un=(y<0).astype(float);rng=np.random.default_rng(2026101103);draws=rng.integers(32,size=(5000,32));rows=[]
ch=[v for v in r['choices'] if v['tie']=='discovery_global_prior']
for c,name in enumerate(r['config']['controller_names']):
 own=[];ou=[]
 for h in ch:
  v=lo[c,:,:,4:] if h['half']==0 else lo[c,:,:,:4];u=un[c,:,:,4:] if h['half']==0 else un[c,:,:,:4];v=v.mean(-1);u=u.mean(-1);choice=np.array(h['state'][c]);own.append(v[np.arange(32),choice]);ou.append(u[np.arange(32),choice])
 a=np.mean(own,0);fixed=lo[c].mean(-1);fu=un[c].mean(-1);b=a[draws].mean(1)-fixed[draws].mean(1).max(1);lower=a[draws].mean(1)-(fixed+fu)[draws].mean(1).max(1)
 rows.append(dict(controller=name,oracle_vs_hindsight_best_fixed=float(a.mean()-fixed.mean(0).max()),simultaneous_fixed_bootstrap95=np.quantile(b,[.025,.975]).tolist(),worst_unknown_simultaneous95=np.quantile(lower,[.025,.975]).tolist()))
write(H/'ALL_FIXED_SENSITIVITY.json',rows);print(rows)
