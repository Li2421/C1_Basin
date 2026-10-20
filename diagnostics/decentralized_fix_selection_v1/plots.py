"""Scientific figures from sealed evidence only."""
import os,sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
import shared_rollout_db
from shared_rollout_db.src import cache_writer
assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
assert (cache_writer.ROOT/'journals').resolve()==ROOT/'shared_rollout_db/journals'
for k,v in [('TMPDIR','tmp'),('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache')]:os.environ[k]=str(HERE/v)
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
 import jax
 jax.config.update('jax_enable_x64',False);assert not jax.config.x64_enabled
except ImportError:pass
assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
assert (cache_writer.ROOT/'journals').resolve()==ROOT/'shared_rollout_db/journals'
def main():
 out=HERE/'figures';out.mkdir(exist_ok=True);r=json.loads((HERE/'results.json').read_text());z=np.load(HERE/'DIAGNOSTIC_FEATURES.npz');x=z['features'];names=z['controller_names'];features=z['feature_names'];groups=['ST','TD','G','DM','AS'];titles=['Offline distilled\nstudents (ST)','Joint MACFlow\nteachers (TD)','Gaussian behavior\ncloning (G)','Distributed model\npredictive control (DM)','A* path\nplanning (AS)'];colors=['#617ca8','#9b78a9','#d5a350','#ba654b','#659383'];featurelabels=['Unfinished freeze proportion','Moving without progress (robot s)','Team progress at 100 steps (m)','Team progress at 200 steps (m)','Team progress at 300 steps (m)','Mean safety intervention (m/s)','Mixed circulation proportion'];fig,axes=plt.subplots(2,4,figsize=(19,9));rng=np.random.default_rng(0)
 for j,ax in enumerate(axes.flat):
  if j>=7:ax.axis('off');continue
  values=[x[[str(n).startswith(f) for n in names],j] for f in groups];ax.boxplot(values,tick_labels=titles,showfliers=False,widths=.5)
  for k,v in enumerate(values):ax.scatter(k+1+rng.uniform(-.13,.13,len(v)),v,s=24,c=colors[k],zorder=3)
  ax.set_title(featurelabels[j]);ax.tick_params(axis='x',labelsize=7);ax.grid(axis='y',alpha=.2)
 fig.suptitle('Outcome-blind diagnostics: 46 frozen controllers, 16 probes each, 300 safety-only steps\nDEVELOPMENT; features frozen before selection');fig.tight_layout()
 for ext in ['png','pdf','svg']:fig.savefig(out/f'diagnostic_distributions.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
 methods=['safety baseline','F13','S500','RHR-A','RHR-B','RHR-C','rule','critic','family_oracle','controller_oracle'];labels=['Safety only; no correction','Constant parameter 13','Parameter 13 for 500 steps, then parameter 10','Right-hand rule A: progress, 1 s, 0.35 m/s','Right-hand rule B: neighbor, 0.8 m, 0.10 m/s','Right-hand rule C: progress, 2 s, 0.25 m/s','Leave-one-family-out rule selector','Leave-one-family-out critic selector','Best single arm per family (oracle)','Best arm per controller (oracle)'];fig,ax=plt.subplots(figsize=(12,7));y=np.arange(len(methods));val=np.array([r['policies'][k]['success_lower']*100 for k in methods]);ci=np.array([r['policies'][k]['ci'] for k in methods])*100;ax.barh(y,val,color=['#aaaaaa']*6+['#336e9d','#b36a36','#8b9871','#8b9871']);ax.errorbar(val,y,xerr=np.maximum(np.stack([val-ci[:,0],ci[:,1]-val]),0),fmt='none',ecolor='black',capsize=3);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,101);ax.set_xlabel('Controller-weighted success (%), paired hierarchical 95% CI');ax.set_title('DEVELOPMENT: five-family transfer of correction selection\n46 controllers × 32 states × two clustered seeds');ax.grid(axis='x',alpha=.2);fig.tight_layout()
 for ext in ['png','pdf','svg']:fig.savefig(out/f'selection_success.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(1,5,figsize=(20,6),sharey=True)
 for f,title,ax in zip(groups,titles,axes):
  v=[r['policies'][k]['families'][f]['success_lower']*100 for k in methods[:8]];ax.bar(np.arange(8),v,color=['#aaaaaa']*6+['#336e9d','#b36a36']);ax.set_xticks(np.arange(8),labels[:8],rotation=90,fontsize=8);ax.set_title(title);ax.set_ylim(0,101);ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel('Success (%)');fig.suptitle('DEVELOPMENT: held-family selections and globally fixed correction arms');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'held_family_success.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
if __name__=='__main__':main()
