from common import *
from audit_utils import source_inventory
import csv

def main():
 names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];states=read(HERE/'A2_ORIGINAL_STATES.json');uids=[s['state_uid']for s in states];refs=read(D/'axis4a_addendum_v1/A2_BASELINE_REFERENCES.json');lo=np.zeros((14,16,2));hi=lo.copy();seen=np.zeros_like(lo,bool)
 for item in refs:
  c=item['case'];r=read(item['raw_path']);i=(names.index(c['base_controller_name']),uids.index(c['state_uid']),['F10','S500'].index(c['arm']));assert not seen[i];seen[i]=True;lo[i]=r['success']if not r['numerical_failure']else 0;hi[i]=lo[i]+r['numerical_failure']
 assert seen.all();rows=read(HERE/'CONTROLLER_RADIUS_CURVES.json')+[dict(controller=n,radius=.9,lower_pct=float(lo[j].mean(0).max()*100),upper_pct=float(hi[j].mean(0).max()*100))for j,n in enumerate(names)];rows.sort(key=lambda r:(names.index(r['controller']),r['radius']));write(HERE/'A_CONTROLLER_RADIUS_CURVES.json',dict(utc=utc(),states=16,futures=1,meaning='Per-controller max of fixed eta10 and500-step13-to10 schedule, original matched16 states, unknown lower/upper bounds; not confidence intervals',rows=rows))
 with(HERE/'A_CONTROLLER_RADIUS_CURVES.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 paths=[HERE/'A_CONTROLLER_RADIUS_CURVES.json',HERE/'A_CONTROLLER_RADIUS_CURVES.csv'];plot_error=None
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,axes=plt.subplots(4,4,figsize=(11,9),sharex=True,sharey=True)
  for j,ax in enumerate(axes.flat):
   if j>=14:ax.axis('off');continue
   q=[r for r in rows if r['controller']==names[j]];x=[r['radius']for r in q];a=[r['lower_pct']for r in q];b=[r['upper_pct']for r in q];ax.plot(x,a,'o-',color='#24588a',lw=1.4,ms=3,label='Known lower');ax.fill_between(x,a,b,color='#24588a',alpha=.25,label='Unknown upper');ax.set_title(f'Ring Flow policy {j+1}',fontsize=10);ax.set_ylim(-2,102);ax.set_xticks([.8,.85,.9,.95,1.]);ax.grid(alpha=.2)
  fig.supxlabel('Obstacle radius (m)');fig.supylabel('Maximum success rate of the two correction methods (%)');fig.suptitle('Same 16 initial states; fixed eta10 or 500-step switch\nShading shows unresolved numerical outcomes, not a confidence interval',fontsize=12);fig.tight_layout(rect=[.02,.02,1,.94])
  for ext in ['pdf','png']:
   dest=HERE/f'A_CONTROLLER_RADIUS_CURVES.{ext}';fig.savefig(dest,dpi=180);paths.append(dest)
  plt.close(fig)
 except ImportError as ex:plot_error=str(ex)
 write(HERE/'A_CURVE_RENDER_STATUS.json',dict(utc=utc(),plot_error=plot_error,controller_index={str(i+1):n for i,n in enumerate(names)},source_data='Original matched16 states; native radius reference uses immutable old rollout labels'));paths.append(HERE/'A_CURVE_RENDER_STATUS.json');seal('A_CONTROLLER_RADIUS_CURVES.receipt.json',paths,new_physical_steps=0);source_inventory('imports_geometry_curves.json')
if __name__=='__main__':main()
