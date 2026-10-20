from core import *
from audit_utils import source_inventory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    grid=read(HERE/'RHR_GRID_RESULTS.json');v=np.array([r['student_mean_lower'] for r in grid['grid']]).reshape(4,3);fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True);im=ax.imshow(v,vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(3),['1 s','2 s','4 s']);ax.set_yticks(range(4),['0.10','0.15','0.25','0.35']);ax.set_xlabel('Consecutive low-speed trigger duration');ax.set_ylabel('Clockwise tangential speed (m/s)')
    for i in range(4):
        for j in range(3):ax.text(j,i,f'{v[i,j]:.1%}',ha='center',va='center',color='white' if v[i,j]<.5 else 'black')
    fig.colorbar(im,ax=ax,label='Equal-student tuning success lower bound');fig.savefig(HERE/'rhr_tuning_grid.png',dpi=180);fig.savefig(HERE/'rhr_tuning_grid.pdf');plt.close(fig);source_inventory('imports_plot_grid.json')
if __name__=='__main__':main()
