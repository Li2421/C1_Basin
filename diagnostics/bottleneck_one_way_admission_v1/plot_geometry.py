"""Optional top-down geometry figure from frozen JSON only."""
import guard
from guard import HERE
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import font_manager

def main():
    font=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family']=font.get_name()
    scenes=json.loads((HERE/'SCENES.json').read_text())['entries'];passages=json.loads((HERE/'PASSAGES.json').read_text())['by_scene']
    chosen=[e for e in scenes if passages[e['scene']['name']]['representative']]
    fig,axes=plt.subplots(5,4,figsize=(16,18),constrained_layout=True)
    for ax,e in zip(axes.flat,chosen):
        sc=e['scene']
        for a,b in sc['walls']:ax.plot([a[0],b[0]],[a[1],b[1]],color='black',linewidth=1)
        for p in passages[sc['name']]['passages']:
            c=np.array(p['centre']);u=np.array(p['axis'])*p['half_length'];v=np.array([-p['axis'][1],p['axis'][0]])*p['width']/2
            ax.add_patch(Polygon([c-u-v,c+u-v,c+u+v,c-u+v],color='royalblue',alpha=.3))
        points=np.array(sc['states'][0]['positions']);ax.scatter(points[:,0],points[:,1],s=8,color='darkred')
        ax.set_title(e['title']+'\n'+str(sc['config']['num_agents'])+'台；通道'+str(len(passages[sc['name']]['passages'])))
        ax.set_aspect('equal');ax.autoscale_view()
    for ax in list(axes.flat)[len(chosen):]:ax.axis('off')
    fig.savefig(HERE/'passages_overview.png',dpi=170);fig.savefig(HERE/'passages_overview.svg');plt.close(fig)

if __name__=='__main__':main()
