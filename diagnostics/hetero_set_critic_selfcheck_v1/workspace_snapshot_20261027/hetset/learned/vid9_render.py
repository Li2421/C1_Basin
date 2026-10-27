"""Render the nine video pairs with plain Chinese titles and plain Chinese file names."""
import os,sys,json,glob,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
from concurrent.futures import ProcessPoolExecutor
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','Droid Sans Fallback'];plt.rcParams['axes.unicode_minus']=False
OUT=sys.argv[1]
def short(m):
 v=m['scene'];p=m['plain']
 return p.replace('，','_').replace('（','_').replace('）','').replace(' ','').replace(',','')
def render(f):
 d=np.load(f,allow_pickle=False);m=json.loads(str(d['meta']));P=d['positions'];G=d['goals'];W=d['walls'];r=float(d['radius']);T,N,_=P.shape
 pts=np.concatenate([P.reshape(-1,2),G]+([W.reshape(-1,2)] if len(W) else []));lo=pts.min(0)-.15;hi=pts.max(0)+.15
 fig,ax=plt.subplots(figsize=(6.6,7.2),dpi=100);ax.set_aspect('equal');ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_xticks([]);ax.set_yticks([])
 for w in W:ax.plot(w[:,0],w[:,1],color='0.25',lw=1.5)
 col=plt.cm.tab20(np.arange(N)%20);ax.scatter(G[:,0],G[:,1],marker='x',c=col,s=40,zorder=2)
 trails=[ax.plot([],[],color=col[i],lw=.8,alpha=.6)[0] for i in range(N)];disc=[ax.add_patch(Circle(P[0,i],r,color=col[i],zorder=3)) for i in range(N)]
 how='只加安全投影（不修正）' if m['role']=='A' else '流水线修正：η = ('+', '.join(f'{x:.2f}' for x in m['eta'])+')'
 ttl=ax.set_title('',fontsize=10,loc='left')
 res={'success':'成功','timeout':'超时（卡住）','collision':'相撞'}[m['outcome']]
 name=f"{m['order']}_{m['family']}_{short(m)}__{'A_只加安全投影_卡住' if m['role']=='A' else 'B_流水线_成功'}.mp4"
 path=os.path.join(OUT,name);wr=FFMpegWriter(fps=20,bitrate=1500);idx=list(range(0,T,2))+[T-1]*20
 with wr.saving(fig,path,dpi=100):
  for t in idx:
   for i in range(N):disc[i].center=P[t,i];trails[i].set_data(P[:t+1,i,0],P[:t+1,i,1])
   end=f'　→ {res}' if t==T-1 else ''
   ttl.set_text(f"{m['family']}：{m['plain']}\ncontroller：{m['controller_name']}\n{how}\n第 {t} / {T-1} 步{end}");wr.grab_frame()
 plt.close(fig);return name
if __name__=='__main__':
 fs=sorted(glob.glob(os.path.join(OUT,'traces','*.npz')))
 with ProcessPoolExecutor(9) as ex:print('\n'.join(sorted(ex.map(render,fs))))
