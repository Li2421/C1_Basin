"""Static scientific figures from saved numerical evidence only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
COL={'TT':'#2864b4','FF':'#d45435'}


def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=190,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def action():
    rows=json.loads((HERE/'raw_G_M.json').read_text())
    q=json.loads((HERE/'fixed_Q_train_test.json').read_text())
    raw=np.load(HERE/'raw_G_M.npz')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for row,(scene,label) in enumerate([('toy_give_way','Toy'),('ring_exchange','Ring')]):
        rr=sorted((r for r in rows if r['scene']==scene),key=lambda x:x['uid'])
        x=np.arange(len(rr))
        ax=axes[row,0]
        for p,shift in [('TT',-.10),('FF',.10)]:
            y=np.array([r['sigma_'+p] for r in rr])
            for k,style in enumerate(['o','s','^']):
                ax.scatter(x+shift,y[:,k],s=14,marker=style,color=COL[p],alpha=.75,
                           label=f'{p} σ{k+1}' if row==0 else None)
        ax.set_yscale('log');ax.set_title(f'{label}: singular values of G')
        ax.set_xlabel('Frozen state index');ax.set_ylabel('singular value')
        ax.grid(alpha=.2)
        bx=axes[row,1]
        for k,p in enumerate(('TT','FF')):
            values=[]
            for r in rr:
                g=raw[f"{r['uid']}_G_{p}"]
                values.append([np.linalg.norm(g[:,j]) for j in range(3)])
            bx.plot(x,np.median(values,axis=0)[0]*np.ones_like(x),alpha=0)
            bx.scatter(x,np.array(values)[:,0],s=17,color=COL[p],marker='o',label=f'{p} basis 1')
            bx.scatter(x,np.array(values)[:,1],s=17,color=COL[p],marker='s',alpha=.7,label=f'{p} basis 2')
            bx.scatter(x,np.array(values)[:,2],s=17,color=COL[p],marker='^',alpha=.7,label=f'{p} basis 3')
        bx.set_yscale('log');bx.set_title(f'{label}: effective basis-direction gains')
        bx.set_xlabel('Frozen state index');bx.set_ylabel('||G e_j||')
        bx.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=6,bbox_to_anchor=(.5,1.06),fontsize=8)
    save(fig,'action_transfer')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
    for ax,(scene,label) in zip(axes,[('toy_give_way','Toy'),('ring_exchange','Ring')]):
        r=q[scene]['test']
        ax.plot(np.arange(12),r['identity_relative_residual'],'o-',label='identity',alpha=.7)
        ax.plot(np.arange(12),r['global_relative_residual'],'s-',label='trained global Q')
        ax.plot(np.arange(12),r['per_state_relative_residual'],'^-',label='per-state best Q',alpha=.8)
        ax.set_title(label+' held-out states');ax.set_xlabel('Held-out state index');ax.set_ylabel('relative G residual');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig,'fixed_Q_heldout')


def trajectory():
    z=np.load(HERE/'trajectory_sensitivity_arrays.npz')
    prot=json.loads((HERE/'closed_loop_protocol.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(10,6.6),constrained_layout=True)
    for ax,st in zip(axes.ravel(),prot['selected_states']):
        uid=st['uid']
        for p in ('TT','FF'):
            s=z[f'{uid}_{p}_zero_0.003_position']
            ax.plot(np.arange(len(s)),np.linalg.norm(s.reshape(len(s),-1),axis=1),color=COL[p],label=p)
        ax.set_title(('Toy' if st['scenario']=='toy_give_way' else 'Ring')+' '+uid[-6:])
        ax.set_xlabel('physical step');ax.set_ylabel('||∂p_t/∂η||F');ax.grid(alpha=.2)
    axes[0,0].legend()
    save(fig,'trajectory_sensitivity')


def boundary():
    normals=json.loads((HERE/'boundary_normals.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for row,uid in enumerate(sorted({r['uid'] for r in normals})):
        d=json.loads((HERE/'boundary_results'/f'{uid}_refined.json').read_text())
        ax=axes[row,0]
        for p in ('TT','FF'):
            rs=d['placements'][p]
            ax.plot([r['fraction'] for r in rs],[r['rho'] for r in rs],'.-',color=COL[p],label=p)
            c=next(x for x in normals if x['uid']==uid and x['chain']==p)
            ax.axvline(c['fraction'],color=COL[p],linestyle='--',alpha=.8)
        ax.axhline(0,color='k',lw=.8)
        ax.set_title('Ring '+uid[-6:]+'; measured line margins')
        ax.set_xlabel('candidate-segment fraction');ax.set_ylabel('ρ (m)');ax.grid(alpha=.2);ax.legend()
        bx=axes[row,1]
        seg=d['segment'];deta=np.asarray(seg['endpoint_eta'][1])-np.asarray(seg['endpoint_eta'][0])
        v=deta/np.linalg.norm(deta)
        w=np.array([0.,1.,0.]);w-=w@v*v;w/=np.linalg.norm(w)
        tt=next(x for x in normals if x['uid']==uid and x['chain']=='TT')
        ff=next(x for x in normals if x['uid']==uid and x['chain']=='FF')
        for c in (tt,ff):
            grad=np.asarray(c['gradient_eps0005']);slope=-float(grad@w)/float(grad@deta)
            yy=np.linspace(-.01,.01,40);xx=c['fraction']+slope*yy
            bx.plot(xx,yy,color=COL[c['chain']],label=c['chain']+' local linear contour')
            bx.scatter([c['fraction']],[0],color=COL[c['chain']],s=30)
        bx.set_title('Measured roots; tangent-plane linearization')
        bx.set_xlabel('candidate-segment fraction');bx.set_ylabel('orthogonal η displacement')
        bx.grid(alpha=.2);bx.legend(fontsize=8)
    save(fig,'local_boundary')


if __name__=='__main__':
    action();trajectory();boundary()
