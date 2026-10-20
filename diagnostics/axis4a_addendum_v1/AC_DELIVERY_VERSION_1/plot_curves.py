"""Plots derived only from stored simulation labels; no environment/controller calls."""
from common import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

def main():
 data=read(HERE/'CONTROLLER_RADIUS_CURVES.json');names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];fig=Figure(figsize=(12,18),layout='constrained');FigureCanvasAgg(fig);axes=fig.subplots(7,2,sharex=True,sharey=True)
 for ax,n in zip(axes.flat,names):
  rows=sorted([q for q in data if q['controller']==n and q['subaxis'].startswith('A2_')],key=lambda q:q['r']);x=[q['r'] for q in rows]
  for key,label,color in [('F10_bounds_pct','F10','#2563eb'),('S500_bounds_pct','S500','#d97706'),('max_bounds_pct','max(F10,S500)','#111827')]:
   y=np.asarray([q[key] for q in rows]);ax.plot(x,y[:,0],marker='o',markersize=3,color=color,label=label,lw=1.2);ax.fill_between(x,y[:,0],y[:,1],alpha=.18,color=color)
  ax.axvline(.9,color='#16a34a',ls=':',lw=1);ax.set_title(n,fontsize=8);ax.set_xticks([.8,.85,.9,.95,1.]);ax.set_ylim(-3,103);ax.grid(alpha=.2);ax.tick_params(labelsize=8)
 axes[0,0].legend(fontsize=7,loc='upper left');fig.supxlabel('Obstacle radius r');fig.supylabel('Success (%) on the same 16 states, seed 0');fig.suptitle('Round37 A2: fixed checkpoint and state cohort across radius\nShading: numerical-unknown identification bounds (not confidence intervals)',fontsize=12);fig.savefig(HERE/'CONTROLLER_RADIUS_CURVES.png',dpi=160);fig.savefig(HERE/'CONTROLLER_RADIUS_CURVES.pdf');print('14controller radius curves plotted',flush=True)
if __name__=='__main__':main()
