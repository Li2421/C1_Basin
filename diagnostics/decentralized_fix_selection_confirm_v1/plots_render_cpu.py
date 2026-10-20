"""Read-only scientific figures, rendered in the existing plotting environment."""
import os,sys,json,hashlib,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
import shared_rollout_db
from shared_rollout_db.src import cache_writer
assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
assert (cache_writer.ROOT/'journals').resolve()==ROOT/'shared_rollout_db/journals'
for key,folder in [('TMPDIR','tmp'),('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache')]:
 p=HERE/folder;p.mkdir(exist_ok=True);os.environ[key]=str(p)
os.environ['JAX_PLATFORMS']='cpu'
def main():
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db') and (cache_writer.ROOT/'journals').resolve()==ROOT/'shared_rollout_db/journals'
 r=json.loads((HERE/'results.json').read_text());new=np.load(HERE/'DIAGNOSTIC_FEATURES.npz');old=np.load(ROOT/'diagnostics/decentralized_fix_selection_v1/DIAGNOSTIC_FEATURES.npz');out=HERE/'figures';out.mkdir(exist_ok=True);groups=['ST','TD','G','DM','AS'];titles=['Distributed\nstudents','Joint MACFlow\nteachers','Gaussian\nbehavior cloning','Distributed model\npredictive control','A*\npath planning'];ftext=['Unfinished freeze proportion','Moving without progress (robot s)','Team progress at step 100 (m)','Team progress at step 200 (m)','Team progress at step 300 (m)','Safety intervention (m/s)','Mixed circulation proportion'];fig,axes=plt.subplots(2,4,figsize=(19,10));rng=np.random.default_rng(1)
 for j,ax in enumerate(axes.flat):
  if j==7:ax.axis('off');continue
  for v,z,color,label in [(0,old,'#939393','Development instances'),(1,new,'#386f9e','New confirmation instances')]:
   values=[z['features'][[str(n).startswith(f) for n in z['controller_names']],j] for f in groups];positions=np.arange(5)*3+v
   bp=ax.boxplot(values,positions=positions,widths=.6,showfliers=False,patch_artist=True)
   for box in bp['boxes']:box.set_facecolor(color);box.set_alpha(.6)
   for k,x in enumerate(values):ax.scatter(positions[k]+rng.uniform(-.15,.15,len(x)),x,s=15,color=color,zorder=3)
  ax.set_xticks(np.arange(5)*3+.5,titles,fontsize=7);ax.set_title(ftext[j]);ax.grid(axis='y',alpha=.2)
 axes.flat[7].text(.02,.8,'Gray: development (round30h)\nBlue: new confirmation instances\n\nSafety-only, 16 probes, 300 steps\nAll new choices frozen before task labels.\n\nStudents: 14 old; 28 new\nTeachers: 14 old; 14 independent seeds\nGaussian: 6 old; 4 new recipes\nModel predictive control: 6 old; 6 new\nA*: 6 old; 6 new',va='top',fontsize=12);fig.suptitle('Outcome-blind diagnostic shift: development versus prospective confirmation');fig.tight_layout()
 for ext in ['png','pdf','svg']:fig.savefig(out/f'new_old_diagnostic_distributions.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
 methods=['safety baseline','F13','S500','RHR-A','RHR-B','RHR-C','primary','secondary','old_critic','family_oracle','controller_oracle'];labels=['Safety only; no correction','Constant parameter 13','Parameter 13 for 500 steps, then parameter 10','Right-hand rule A: strong, 1 s, 0.35 m/s','Right-hand rule B: weak, neighbor 0.8 m','Right-hand rule C: progress, 2 s, 0.25 m/s','Frozen primary three-branch selector','Frozen two-level selector (exploratory)','Frozen universal critic (original menu)','Best single arm per family (oracle)','Best arm per controller (oracle)'];val=np.array([r['policies'][k]['success_lower'] for k in methods])*100;ci=np.array([r['policies'][k]['CI'] for k in methods])*100;fig,ax=plt.subplots(figsize=(13,8));yy=np.arange(len(methods));ax.barh(yy,val,color=['#aaaaaa']*6+['#336e9d','#d39842','#a282a7','#8b9871','#8b9871']);ax.errorbar(val,yy,xerr=np.maximum(np.stack([val-ci[:,0],ci[:,1]-val]),0),fmt='none',ecolor='black',capsize=3);ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set_xlim(0,101);ax.set_xlabel('Controller-weighted success (%); paired hierarchical 95% CI');ax.set_title('Prospective confirmation: 58 frozen instances, 32 fresh states, two clustered seeds\nPrimary comparison uses six fixed correction arms; two-level selector is exploratory');ax.grid(axis='x',alpha=.2);fig.tight_layout()
 for ext in ['png','pdf','svg']:fig.savefig(out/f'selector_success.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(1,5,figsize=(21,7),sharey=True)
 for f,title,ax in zip(groups,titles,axes):
  m=methods[:8];v=[r['policies'][k]['families'][f]['success_lower']*100 for k in m];ax.bar(np.arange(8),v,color=['#aaa']*6+['#336e9d','#d39842']);ax.set_xticks(np.arange(8),labels[:8],rotation=90,fontsize=8);ax.set_title(title);ax.set_ylim(0,101);ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel('Success (%)');fig.suptitle('Frozen correction selection on each new controller family');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'family_success.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig);sources={}
 for m in list(sys.modules.values()):
  p=getattr(m,'__file__',None)
  if p and Path(p).suffix=='.py' and Path(p).is_file():sources[str(Path(p).resolve())]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
 (HERE/'imports_plots.json').write_text(json.dumps(dict(imported_sources=sources,jax_imported=False,jax_enable_x64=None,precision_context='Pure NumPy/matplotlib rendering; parent finalization process restores JAX x64=False; no controller, inference, or statistics executed here.'),indent=2)+'\n')
if __name__=='__main__':main()
