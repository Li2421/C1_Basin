"""Standalone figures rendered solely from frozen scene and result artifacts."""
import guard
from guard import HERE
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import font_manager

def main():
    fonts=[p for p in font_manager.findSystemFonts()if any(k in p.lower()for k in('notosanscjk','wqy','droidsansfallback'))]
    if fonts:plt.rcParams['font.family']=font_manager.FontProperties(fname=fonts[0]).get_name()
    plt.rcParams['svg.fonttype']='none'
    frozen=json.loads((HERE/'SCENES_FROZEN.json').read_text());fig,axes=plt.subplots(5,4,figsize=(15,18),constrained_layout=True)
    for ax,e in zip(axes.flat,frozen['entries']):
        ax.set_title(e['title'],fontsize=10)
        if 'scene'not in e:
            ax.text(.5,.5,'几何规格不可实现\n未运行',ha='center',va='center',transform=ax.transAxes);ax.axis('off');continue
        s=e['scene'];state=s['states'][0];p=np.asarray(state['positions']);g=np.asarray(state['goals']);r=s['config']['agent_radius'];colors=plt.cm.tab20(np.linspace(0,1,len(p)))
        for a,b in np.asarray(s['walls']).reshape(-1,2,2):ax.plot([a[0],b[0]],[a[1],b[1]],color='black',linewidth=1.4)
        for i,(a,b)in enumerate(zip(p,g)):
            ax.add_patch(Circle(a,r,facecolor=colors[i],edgecolor='black',linewidth=.4))
            ax.scatter(*b,color=colors[i],marker='x',s=22)
            path=np.asarray(s['initial_paths'][0][i]);ax.plot(path[:,0],path[:,1],color=colors[i],alpha=.35,linewidth=.7)
        bounds=s['bounds'];span=np.vstack([p,g]);lo=np.minimum(span.min(0)-.2,bounds[:2]);hi=np.maximum(span.max(0)+.2,bounds[2:])
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.tick_params(labelsize=7)
        ax.set_xlabel(f"N={len(p)}, r={r}m, route={s['route_guidance']}",fontsize=8)
    for ax in list(axes.flat)[len(frozen['entries']):]:ax.axis('off')
    fig.savefig(HERE/'scene_overview.svg');fig.savefig(HERE/'scene_overview.png',dpi=160);plt.close(fig)
    if(HERE/'SCENE_DESCRIPTORS.json').exists():
        data=json.loads((HERE/'SCENE_DESCRIPTORS.json').read_text());matrix=np.array([[np.nan if x is None else x for x in row]for row in data['distance_matrix']])
        fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(matrix,cmap='viridis');fig.colorbar(im,ax=ax,label='Standardized feature distance');ax.set_xticks(range(17),range(1,18));ax.set_yticks(range(17),range(1,18));ax.set_title('Geometry and traffic descriptor distances');fig.tight_layout();fig.savefig(HERE/'scene_distances.svg');plt.close(fig)

if __name__=='__main__':main()
