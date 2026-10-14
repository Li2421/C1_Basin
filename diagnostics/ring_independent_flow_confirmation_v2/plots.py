from core import *
prereg();labels_allowed();r=read(HERE/'results.json');p=read(HERE/'protocol.json')
sys.path.insert(0,str(R9/'vendor'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
with (HERE/'ARM_RATES.csv').open() as f:ars=list(csv.DictReader(f))
fig,axs=plt.subplots(7,2,figsize=(13,24),sharey=True)
for ax,n in zip(axs.flat,p['controller_order']):
    items=[x for x in ars if x['controller']==n];ys=np.array([float(x['success_lower']) for x in items])*100;low=np.array([float(x['CI95_lower']) for x in items])*100;high=np.array([float(x['CI95_upper']) for x in items])*100;ax.bar(p['arms'],ys,color=['#4979a7','#bf6574','#84a59d','#5b9f7b','#bb985a','#9176a8']);ax.vlines(np.arange(6),low,high,color='black');ax.plot(np.arange(6),low,'_',color='black');ax.plot(np.arange(6),high,'_',color='black');ax.set_title(n.replace('_macflow__independent',''));ax.set_ylim(0,104);ax.set_ylabel('Success lower point (%)');ax.grid(axis='y',alpha=.2)
fig.suptitle('Round14A: fourteen independently trained policies, six arms\n95% state-cluster CI; unknowns retained separately');fig.tight_layout();fig.savefig(HERE/'six_arm_success.png',dpi=180);fig.savefig(HERE/'six_arm_success.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,5))
for ax,scheme in zip(axs,['primary','secondary']):
    m=r['prediction_metrics'][scheme];x=np.array(m['actual_E'])*100;y=np.array(m['predictions'])*100;ax.scatter(x,y,color='#4979a7');span=[min(x.min(),y.min(),0)-2,max(x.max(),y.max())+2];ax.plot(span,span,'--',color='gray');ax.set(xlabel='Actual E (pp)',ylabel='Frozen Ehat (pp)',title=f"{scheme}: rho={m['Spearman']:.3f}, RMSE ratio={m['RMSE_ratio']:.3f}" if m['Spearman'] is not None and m['RMSE_ratio'] is not None else scheme)
    for i,n in enumerate(p['controller_order']):
        label=n.replace('base_u_','').replace('_macflow__independent','').replace('plus_w_v4_train_targeted','w4-targeted').replace('plus_w_','w-').replace('_local','L')
        right=x[i]>40;offset=(-4,-12) if 'targeted' in label else (4,-12) if label=='v8L' else (-4,5) if right else (4,5)
        ax.annotate(label,(x[i],y[i]),fontsize=8,xytext=offset,textcoords='offset points',ha='right' if right else 'left')
    ax.set_xlim(span);ax.set_ylim(span)
    ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(HERE/'Ehat_vs_E.png',dpi=180);fig.savefig(HERE/'Ehat_vs_E.pdf');plt.close(fig)
