"""Nested 3/5/9 cubical homology and margin-threshold stability."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label,generate_binary_structure
from analyze_panel import betti_from_voxels

HERE=Path(__file__).resolve().parent


def main():
    protocol=json.loads((HERE/'grid9_protocol.json').read_text())
    out=[];arrays={}
    for st in protocol['states']:
        uid=st['uid']
        for chain in ('TT','FF'):
            rho=np.empty((9,9,9));term=np.empty((9,9,9),dtype=object);success=np.empty((9,9,9),dtype=bool)
            for i in range(9):
                for j in range(9):
                    for k in range(9):
                        if i%2==j%2==k%2==0:
                            m=HERE/'panel_results'/f'{uid}_grid5_{(i//2)*25+(j//2)*5+k//2:03d}_{chain}.json'
                        else:m=HERE/'grid9_results'/f'{uid}_{chain}_{i}{j}{k}.json'
                        r=json.loads(m.read_text());rho[i,j,k]=r['rho'] if r['rho'] is not None else -np.inf
                        term[i,j,k]=r['termination'];success[i,j,k]=bool(r['success'])
            arrays[(uid,chain)]=rho
            for res,step in ((3,4),(5,2),(9,1)):
                x=rho[::step,::step,::step];failure=(term[::step,::step,::step]=='numerical_failure')
                for threshold in (-.02,-.01,-.005,0.,.005,.01,.02):
                    mask=success[::step,::step,::step] if threshold==0. else x>threshold
                    hom=betti_from_voxels(mask)
                    _,b0_6=label(mask,generate_binary_structure(3,1))
                    _,b0_26=label(mask,generate_binary_structure(3,3))
                    edge=np.zeros_like(mask,dtype=bool)
                    for ax in range(3):
                        lower=[slice(None)]*3;lower[ax]=0;edge[tuple(lower)]=True
                        upper=[slice(None)]*3;upper[ax]=-1;edge[tuple(upper)]=True
                    out.append(dict(uid=uid,scene=st['scenario'],chain=chain,resolution=res,
                        threshold=threshold,success_voxels=int(mask.sum()),numerical_failure_voxels=int(failure.sum()),
                        success_voxels_touching_box=int((mask&edge).sum()),
                        component_counts_6_26=[int(b0_6),int(b0_26)],**hom))
                ids=np.argwhere(failure);base=success[::step,::step,::step]
                variants=[]
                exhaustive=len(ids)<=(5 if res==9 else 10)
                if exhaustive:
                    for bits in range(1<<len(ids)):
                        mask=base.copy()
                        for k,ix in enumerate(ids):mask[tuple(ix)]=bool((bits>>k)&1)
                        variants.append(tuple(betti_from_voxels(mask)['beta']))
                else:
                    variants=[tuple(betti_from_voxels(base)['beta']),tuple(betti_from_voxels(base|failure)['beta'])]
                out.append(dict(uid=uid,scene=st['scenario'],chain=chain,resolution=res,
                    threshold='numerical_failure_sensitivity',unknown_voxels=int(len(ids)),exhaustive=exhaustive,
                    possible_betti=sorted(set(variants))))
    result=dict(rows=out,
        interpretation='Homology of closed success voxels at sampled centers, clipped observation box. Features touching box boundary or failing resolution/threshold stability are not topology evidence.',
        filtration='multi-threshold rho sweep, not full persistence barcode; rho undefined at numerical failure')
    (HERE/'grid9_topology.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(HERE/'grid9_rho_arrays.npz',**{uid+'_'+ch:x for (uid,ch),x in arrays.items()})
    fig,axes=plt.subplots(2,2,figsize=(9,8))
    for ai,st in enumerate(protocol['states']):
        for aj,ch in enumerate(('TT','FF')):
            x=arrays[(st['uid'],ch)]
            axes[ai,aj].imshow((x[:,:,4]>0).T,origin='lower',cmap='Greys',vmin=0,vmax=1)
            axes[ai,aj].set_title(st['uid'].split('_')[-1]+' '+ch+' η3 mid-slice')
            axes[ai,aj].set_xlabel('η1 grid index');axes[ai,aj].set_ylabel('η2 grid index')
    fig.tight_layout();(HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures/grid9_mid_slices.pdf');fig.savefig(HERE/'figures/grid9_mid_slices.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
    for si,st in enumerate(protocol['states']):
        for bi in range(3):
            ax=axes[si,bi]
            for ch,color in (('TT','#1f77b4'),('FF','#d62728')):
                for res,style in ((3,':'),(5,'--'),(9,'-')):
                    rr=sorted((r for r in out if r['uid']==st['uid'] and r['chain']==ch and
                               r['resolution']==res and isinstance(r['threshold'],float)),key=lambda r:r['threshold'])
                    ax.plot([r['threshold'] for r in rr],[r['beta'][bi] for r in rr],
                            color=color,linestyle=style,marker='.',label=f'{ch} {res}³' if si==bi==0 else None)
            ax.set_title(st['uid'].split('_')[-1]+f' β{bi}')
            ax.set_ylabel('Betti number')
            if si==1:ax.set_xlabel('diagnostic rho threshold')
    axes[0,0].legend(fontsize=7);fig.tight_layout()
    fig.savefig(HERE/'figures/topology_betti.pdf');fig.savefig(HERE/'figures/topology_betti.png',dpi=180);plt.close(fig)
    print(json.dumps([r for r in out if r['threshold']==0.],indent=2))


if __name__=='__main__':main()
