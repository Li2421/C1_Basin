"""Read simulation/training evidence only, render frozen training curves."""
import pathlib,json,os
H=pathlib.Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(H/'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 out=H/'training_curves';out.mkdir(exist_ok=True)
 for p in sorted((H/'students').glob('*/*/*_history.json')):
  rows=json.loads(p.read_text());fig,ax=plt.subplots(figsize=(6,3));ax.plot([r['step'] for r in rows],[r['dev_mse'] for r in rows],label='Fixed development MSE');ax.set(xlabel='Training step',ylabel='Mean squared error',title='/'.join(p.parts[-3:]));ax.legend();fig.tight_layout();fig.savefig(out/('_'.join(p.parts[-3:]).replace('.json','.png')),dpi=150);plt.close(fig)
if __name__=='__main__':main()
