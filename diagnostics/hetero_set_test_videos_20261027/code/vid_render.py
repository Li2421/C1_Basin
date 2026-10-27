import sys,os,json,glob,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
from concurrent.futures import ProcessPoolExecutor
OUT=sys.argv[1]
CN={'centralized_sampling':'centralized sampling MPC','gradient_mpc':'gradient-descent MPC','distributed_shooting':'distributed MPC (shooting)','centralized_mppi':'centralized MPPI','cross_entropy':'cross-entropy','distributed_cross_entropy':'distributed cross-entropy','conflict_based_search':'conflict-based search','prioritized_astar':'prioritized A*'}
CORR={'pipeline':'pipeline eta (generator 16 proposals + critic)','best_fixed_eta':'best fixed eta sE22','eta13':'fixed eta13'}
def render(f):
 d=np.load(f,allow_pickle=False);m=json.loads(str(d['meta']));P=d['positions'];G=d['goals'];W=d['walls'];r=float(d['radius']);T,N,_=P.shape
 pts=np.concatenate([P.reshape(-1,2),G]+([W.reshape(-1,2)] if len(W) else []));lo=pts.min(0)-.15;hi=pts.max(0)+.15
 fig,ax=plt.subplots(figsize=(6.4,6.8),dpi=100);ax.set_aspect('equal');ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_xticks([]);ax.set_yticks([])
 for w in W:ax.plot(w[:,0],w[:,1],color='0.25',lw=1.5)
 col=plt.cm.tab20(np.arange(N)%20)
 ax.scatter(G[:,0],G[:,1],marker='x',c=col,s=40,zorder=2)
 trails=[ax.plot([],[],color=col[i],lw=.8,alpha=.6)[0] for i in range(N)];disc=[ax.add_patch(Circle(P[0,i],r,color=col[i],zorder=3)) for i in range(N)]
 role='SAFETY ONLY (eta = 0)' if m['role']=='safety_failure' else 'CORRECTION: '+CORR[m['correction']]
 eta=', '.join(f'{x:.2f}' for x in m['eta'])
 ttl=ax.set_title('',fontsize=9,loc='left')
 name=os.path.basename(f)[:-4];sub=os.path.join(OUT,m['scene']);os.makedirs(sub,exist_ok=True);path=os.path.join(sub,f"{m['controller_type']}__{m['role']}.mp4")
 wr=FFMpegWriter(fps=20,bitrate=1500);idx=list(range(0,T,2))+[T-1]*20
 with wr.saving(fig,path,dpi=100):
  for t in idx:
   for i in range(N):disc[i].center=P[t,i];trails[i].set_data(P[:t+1,i,0],P[:t+1,i,1])
   end=f"  ->  {m['outcome'].upper()}" if t==T-1 else ''
   ttl.set_text(f"{m['scene']}  |  state {m['state']}\n{CN[m['controller_type']]}\n{role}  eta=({eta})\nstep {t}/{T-1}{end}")
   wr.grab_frame()
 plt.close(fig);return path
if __name__=='__main__':
 fs=sorted(glob.glob(OUT+'/traces/*.npz'))
 with ProcessPoolExecutor(18) as ex:out=list(ex.map(render,fs))
 print(len(out),'videos')
