"""Standalone scientific plots; no simulation or model queries."""
import os,sys,json,hashlib
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
 r=json.loads((HERE/'results.json').read_text());out=HERE/'figures';out.mkdir(exist_ok=True)
 names={'DM':'Distributed model predictive control (DM)','AS':'A* path planning (AS; inconclusive)','G':'Gaussian behavior cloning (G)','V0':'Offline distilled students (V0; development)'}
 methods=['nominal only','safety baseline','F10','F13','S300','S500','RHR-fair','A3_full','oracle']
 labels=['Controller only','Safety only','Constant parameter 10 (F10)','Constant parameter 13 (F13)','Parameter 13 for 300 steps (S300)','Parameter 13 for 500 steps (S500)','Family-tuned right-hand rule','Frozen universal critic','Best of four correction arms']
 fig,axes=plt.subplots(2,2,figsize=(14,11),sharex=True,sharey=True)
 for ax,f in zip(axes.flat,names):
  g=r['groups'][f]
  if not g.get('controllers'):ax.set_title(names[f]);continue
  vals=[(g['arms'] if a in g['arms'] else g['choices'])[a]['success'] for a in methods];y=np.array([v['lower']['estimate'] for v in vals])*100;lo=np.array([v['CI_unknown_envelope'][0] for v in vals])*100;hi=np.array([v['CI_unknown_envelope'][1] for v in vals])*100;x=np.arange(len(y));colors=['#a7a7a7','#636363','#5b8db8','#7ca9cd','#d7a94b','#c67425','#8ca66b','#8766ad','#5f7790'];ax.barh(x,y,color=colors);ax.errorbar(y,x,xerr=np.array([y-lo,hi-y]),fmt='none',ecolor='black',capsize=2,lw=.8);ax.set_yticks(x,labels);ax.invert_yaxis();ax.set_xlim(0,101);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.set_title(names[f]+'\n'+str(len(g['controllers']))+' controllers, 32 states, two seeds');ax.set_xlabel('Success rate (%); hierarchical 95% CI')
 fig.suptitle('Frozen correction schedules across black-box controller families',fontsize=15);fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'family_success.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5));colors=['#276897','#ce862f'];y=0
 for f in ['DM','AS','G']:
  g=r['groups'][f]
  if not g.get('controllers'):continue
  for k,label,color in [('one','Schedule minus safety',colors[0]),('two','Schedule minus tuned right-hand rule',colors[1])]:
   e=g['endpoints'][k];v=e['lower']['estimate']*100;lo,hi=np.array(e['CI_unknown_envelope'])*100;ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color=color,capsize=4);ax.text(hi+1,y,e['decision'],va='center',fontsize=9);y+=1
 ax.set_yticks(np.arange(y),[names[f]+'\n'+label for f in ['DM','AS','G'] if r['groups'][f].get('controllers') for label in ['Schedule minus safety','Schedule minus tuned right-hand rule']]);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Success difference (percentage points); paired hierarchical 95% CI');ax.set_title('Pre-registered family endpoints; unknown-outcome conservative envelope');ax.grid(axis='x',alpha=.2);fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'family_endpoints.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig)
 sources={}
 for m in sys.modules.values():
  p=getattr(m,'__file__',None)
  if p and Path(p).suffix=='.py' and Path(p).is_file():sources[str(Path(p).resolve())]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
 (HERE/'imports_plots.json').write_text(json.dumps(dict(imported_sources=sources,jax_enable_x64=bool(sys.modules['jax'].config.x64_enabled) if 'jax' in sys.modules else None,jax_not_imported='jax' not in sys.modules,no_model_queries=True,db_package=shared_rollout_db.__file__,default_journals=str(cache_writer.ROOT/'journals')),indent=2)+'\n')
 print('PLOTS PASS')
if __name__=='__main__':main()
