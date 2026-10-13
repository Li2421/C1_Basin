"""Static scientific summary figures from saved, verified diagnostics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def main():
    actions=json.loads((HERE/'action_rows.json').read_text())
    cached=json.loads((HERE/'cached_geometry_rows.json').read_text())
    global_rows=json.loads((HERE/'global_geometry_rows.json').read_text())
    geometry=json.loads((HERE/'local_geometry_properties.json').read_text())
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    for scene,style in (('ring_exchange','-'),('toy_give_way','--')):
        for chain,color in (('TT','#1f77b4'),('FF','#d95f02')):
            aa=[r for r in actions if r['scene']==scene and r['chain']==chain]
            xs=np.asarray(aa[0]['eps'])
            ys=[np.mean([r['branch_stable_axis_count'][i]==3 for r in aa]) for i in range(len(xs))]
            axs[0,0].plot(xs,ys,style,color=color,marker='o',label=f'{scene} {chain}')
    axs[0,0].set_xscale('log');axs[0,0].invert_xaxis();axs[0,0].set_ylim(-.03,1.03)
    axs[0,0].set_xlabel('eta perturbation epsilon');axs[0,0].set_ylabel('fixed projection signature fraction')
    axs[0,0].legend(fontsize=7)
    for scene,color in (('ring_exchange','#1f77b4'),('toy_give_way','#d95f02')):
        aa=[r for r in cached if r['scene']==scene]
        axs[0,1].scatter([r['B_near_null_ratio'] for r in aa],
            [r['chains']['FF']['near_null_ratio'] for r in aa],label=scene,color=color,alpha=.75)
    axs[0,1].set_xscale('log');axs[0,1].set_yscale('log')
    axs[0,1].plot([1e-5,1],[1e-5,1],'k:',lw=.7)
    axs[0,1].set_xlabel('$\\sigma_3(B)/\\sigma_1(B)$')
    axs[0,1].set_ylabel('$\\sigma_3(G_{FF})/\\sigma_1(G_{FF})$');axs[0,1].legend(fontsize=8)
    for scene,color in (('ring_exchange','#1f77b4'),('toy_give_way','#d95f02')):
        aa=[r for r in global_rows if r['scene']==scene]
        axs[1,0].scatter([r['loss_fraction'] for r in aa],[r['gain_fraction'] for r in aa],label=scene,color=color)
    axs[1,0].plot([0,.6],[0,.6],'k:',lw=.8)
    axs[1,0].set_xlim(0,.6);axs[1,0].set_ylim(0,.6)
    axs[1,0].set_xlabel('TT-only loss fraction');axs[1,0].set_ylabel('FF-only gain fraction')
    axs[1,0].legend(fontsize=8)
    stable=[r for r in geometry['pairs'] if r['both_curvature_regular']]
    labels=[r['uid'][-6:] for r in stable];xs=np.arange(len(labels))
    axs[1,1].bar(xs-.18,[max(abs(x) for x in r['TT']['k_median']) for r in stable],.34,label='TT',color='#1f77b4')
    axs[1,1].bar(xs+.18,[max(abs(x) for x in r['FF']['k_median']) for r in stable],.34,label='FF',color='#d95f02')
    axs[1,1].set_xticks(xs,labels);axs[1,1].set_ylabel('max absolute principal curvature')
    axs[1,1].set_xlabel('regular Ring boundary UID');axs[1,1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE/'figures/property_overview.pdf')
    fig.savefig(HERE/'figures/property_overview.png',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
