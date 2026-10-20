from common import *
from audit_utils import source_inventory
sys.path.insert(0,str(D/'controller_loco_real_flow_v1/plot_vendor'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 data=read(HERE/'A_CONTROLLER_RADIUS_CURVES.json');rows=data['rows'];names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];fig,axes=plt.subplots(4,4,figsize=(11,9),sharex=True,sharey=True)
 for j,ax in enumerate(axes.flat):
  if j>=14:ax.axis('off');continue
  q=[r for r in rows if r['controller']==names[j]];x=[r['radius']for r in q];a=[r['lower_pct']for r in q];b=[r['upper_pct']for r in q];ax.plot(x,a,'o-',color='#24588a',lw=1.4,ms=3,label='Known lower bound');ax.fill_between(x,a,b,color='#24588a',alpha=.25,label='Unknown upper bound');ax.set_title(f'Ring Flow policy {j+1}',fontsize=10);ax.set_ylim(-2,102);ax.set_xticks([.8,.85,.9,.95,1.]);ax.grid(alpha=.2)
 fig.supxlabel('Obstacle radius (m)');fig.supylabel('Maximum success rate of the two correction methods (%)');fig.suptitle('Same 16 initial states; fixed eta10 or 500-step switch\nShading shows unresolved numerical outcomes, not a confidence interval',fontsize=12);fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower right',ncol=1,fontsize=9,bbox_to_anchor=(.95,.10));fig.tight_layout(rect=[.02,.02,1,.94]);paths=[]
 for ext in ['pdf','png','svg']:
  dest=HERE/f'A_CONTROLLER_RADIUS_CURVES.{ext}';fig.savefig(dest,dpi=180);paths.append(dest)
 plt.close(fig);write(HERE/'A_CURVE_PLOTS_STATUS.json',dict(utc=utc(),passed=True,matplotlib_source=matplotlib.__file__,library_read_only=True,package_installed=False,new_physical_steps=0,source_data_sha256=sha(HERE/'A_CONTROLLER_RADIUS_CURVES.json')));source_inventory('imports_geometry_plot_readonly.json');seal('A_CURVE_PLOTS.receipt.json',[*paths,HERE/'A_CURVE_PLOTS_STATUS.json',HERE/'A_CONTROLLER_RADIUS_CURVES.json',HERE/'geometry_plot_readonly.py'],new_physical_steps=0,read_only_existing_plot_library=True)
if __name__=='__main__':main()
