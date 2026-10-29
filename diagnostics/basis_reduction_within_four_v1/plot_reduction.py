"""Optional reduction graph, strictly from frozen combination/result JSON."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def main():
 d=json.loads((HERE/'REDUCTION.json').read_text());cs=json.loads((HERE/'COMBINATIONS.json').read_text())['combinations'];rows={r['code']:r for r in d['rows']};positions={}
 fig,ax=plt.subplots(figsize=(13,8))
 for dim in(5,4,3,2):
  items=[c for c in cs if c['dimension']==dim]
  for i,c in enumerate(items):
   x=i-(len(items)-1)/2;y=dim;positions[c['code']]=(x,y);r=rows[c['code']];color='#2a9d8f'if r['qualified']else'#aaaaaa'
   ax.scatter(x,y,c=color,s=180,zorder=3);ax.text(x,y+.1,f"{c['code']}\n{r['coverage']['64']}/272",ha='center',fontsize=8)
 for c in cs:
  for p in(c['parents']or(['reference']if c['dimension']==4 else[])):
   if p in positions:ax.annotate('',xy=positions[c['code']],xytext=positions[p],arrowprops=dict(arrowstyle='->',color='#bbbbbb',lw=.8))
 ax.set_yticks([2,3,4,5]);ax.set_ylabel('Free coefficients');ax.set_xticks([]);ax.set_title('Frozen reduction paths (green: meets coverage criterion)');ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(HERE/'reduction_paths.png',dpi=180);fig.savefig(HERE/'reduction_paths.svg')

if __name__=='__main__':main()
