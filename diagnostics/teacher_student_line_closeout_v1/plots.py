"""Static scientific figures from saved analysis outputs; no simulation imports."""
import os,pathlib,json,importlib.util
H=pathlib.Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(H/'mplconfig');os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
if importlib.util.find_spec('jax'):
 import jax
 jax.config.update('jax_enable_x64',False)
precision={'numpy_dtype':'float64','JAX_imported':bool(importlib.util.find_spec('jax')),'jax_enable_x64':False if importlib.util.find_spec('jax') else 'not used in plotting environment'}
OUT=H/'figures';OUT.mkdir(exist_ok=True)
read=lambda n:json.loads((H/n).read_text())
N={'TD':'Joint teacher\n(distributed safety)','TC':'Joint teacher\n(centralized safety)','V0':'Offline distilled\nstudent (V0)','V1':'Closed-loop relabeled\nstudent (V1)','V3':'Shared-noise relabeled\nstudent (V3)','Z':'Zero-action\ncontroller (Z)','expert':'Centralized expert\n(privileged reference)'}
COL={'TD':'#607d8b','V0':'#0072b2','V1':'#d55e00','V3':'#009e73','Z':'#8e44ad','expert':'#222222'}
A={'safety baseline':'Safety layer only','F10':'Fixed correction 10','F13':'Fixed correction 13','S300':'Correction 13 to10 at300','S500':'Correction 13 to10 at500','nominal only':'Controller only','RHR-fair':'Fairly tuned right-hand rule','critic':'Frozen general critic','oracle':'Per-controller post-hoc best'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=170);fig.savefig(OUT/(name+'.pdf'));plt.close(fig)
def point(r):return np.nan if r is None or r.get('estimate') is None else r['estimate']
def bars(ax,labels,values,errs=None):
 x=np.arange(len(labels));ax.bar(x,values,color=[COL.get(g,'#607d8b') for g in labels]);ax.set_xticks(x,[N.get(g,g) for g in labels]);
 if errs is not None:ax.errorbar(x,values,yerr=np.asarray(errs).T,fmt='none',ecolor='black',capsize=3)
def main():
 a1=read('analysis1_teacher_ceiling.json');a2=read('analysis2_nominal_failure.json');a3=read('analysis3_safety_stalls.json');a4=read('analysis4_correction.json');a5=read('analysis5_zero_controller.json');a6=read('analysis6_relabeling_shared_noise.json');p=read('analysis7_paper_cells.json');curve=read('PLOT_DATA_FINAL.json');sec=read('SECONDARY_ANALYSES.json');groups=['TD','V0','V1','V3']
 fig,axs=plt.subplots(1,3,figsize=(12,3.7),sharex=True,sharey=True)
 for ax,v in zip(axs,['V0','V1','V3']):
  for frame,marker in [('global','x'),('local','o')]:
   rr=[r for r in a1['per_teacher'] if r['frame']==frame];ax.scatter([r['teacher_nominal']*100 for r in rr],[r[v]*100 for r in rr],label=frame+'-frame teacher',marker=marker,color=COL[v],s=50)
  ax.plot([0,100],[0,100],color='gray',ls='--',lw=1);ax.set(xlim=(-3,103),ylim=(-3,103),xlabel='Teacher controller-only success (%)',title=N[v].replace('\n',' '));ax.text(.04,.94,'Pearson r='+f"{a1['comparisons'][v]['Pearson']:.3f}"+'\n10 global teachers overlap at origin',transform=ax.transAxes,va='top',fontsize=8)
 axs[0].set_ylabel('Student controller-only success (%)');axs[-1].legend(loc='lower right',fontsize=7);fig.suptitle('Analysis 1: teacher/student association; global subgroup has zero variance',y=1.03);fig.tight_layout();save(fig,'analysis1_teacher_student_scatter')
 fig,axs=plt.subplots(1,2,figsize=(11,4.4));labels=['stationary','clockwise consistent','counterclockwise consistent','mixed'];colors=['#bdbdbd','#56b4e9','#009e73','#d55e00'];bottom=np.zeros(4)
 for j,l in enumerate(labels):
  val=np.array([a2['groups'][g]['mode_counts'][str(j)]/896*100 for g in groups]);axs[0].bar(np.arange(4),val,bottom=bottom,label=l,color=colors[j]);bottom+=val
 axs[0].set_xticks(np.arange(4),[N[g] for g in groups]);axs[0].set(ylabel='Recorded episodes (%)',title='Controller-only circulation');axs[0].legend(fontsize=7,loc='upper left',bbox_to_anchor=(0,1.02));x=np.arange(4);axs[1].bar(x-.18,[point(a2['groups'][g]['mixed_failure'])*100 for g in groups],.35,label='Among known failures',color='#d55e00');axs[1].bar(x+.18,[point(a2['groups'][g]['mixed_success'])*100 for g in groups],.35,label='Among successes',color='#0072b2');axs[1].set_xticks(x,[N[g] for g in groups]);axs[1].set(ylabel='Mixed circulation (%)',title='Mixed direction is a minority of failures');axs[1].legend(fontsize=8);fig.tight_layout();save(fig,'analysis2_nominal_modes')
 fig,axs=plt.subplots(2,3,figsize=(13,7));ax=axs[0,0]
 for g in groups:
  values=[r['last_progress_seconds'] for r in a3['timeout_rows'] if r['group']==g];ax.hist(values,bins=np.linspace(0,35,15),histtype='step',density=True,label=N[g].replace('\n',' '),color=COL[g])
 ax.set(xlabel='Last >=0.05m aggregate progress mark (s)',ylabel='Conditional density',title='Safety-only timeout timing');ax.legend(fontsize=6)
 ax=axs[0,1];x=np.arange(4);ax.bar(x-.18,[point(a3['groups'][g]['stall_robot_seconds']) for g in groups],.35,label='Low-speed stalls >=2s',color='#56b4e9');ax.bar(x+.18,[point(a3['groups'][g]['moving_no_progress_robot_seconds']) for g in groups],.35,label='Moving without2s progress',color='#d55e00');ax.set_xticks(x,[N[g] for g in groups],fontsize=7);ax.set(ylabel='Mean robot-seconds per timeout',title='Safety does not usually stop all motion');ax.legend(fontsize=6)
 for ax,g in zip([axs[0,2],*axs[1]],groups):
  rr=[r for r in a3['timeout_rows'] if r['group']==g];xy=np.asarray([r['final_positions'] for r in rr]).reshape(-1,2);ax.scatter(xy[:,0],xy[:,1],s=1,alpha=.18,color=COL[g]);ax.add_patch(Circle((0,0),.9,color='#444444'));ax.add_patch(Circle((0,0),3.25,fill=False,color='black'));ax.add_patch(Circle((0,0),1.065,fill=False,ls=':',color='gray'));ax.add_patch(Circle((0,0),3.085,fill=False,ls=':',color='gray'));ax.set(aspect='equal',xlim=(-3.4,3.4),ylim=(-3.4,3.4),title=N[g].replace('\n',' '),xlabel='x (m)',ylabel='y (m)')
 fig.suptitle('Analysis 3: timing and observed final robot positions of safety-only timeouts',y=1.01);fig.tight_layout();save(fig,'analysis3_stall_locations_timing')
 fig,axs=plt.subplots(1,3,figsize=(14,4.3))
 for g in groups:
  x=np.asarray(curve[g]['S500']['steps']);axs[0].plot(x,curve[g]['S500']['mean_progress'],label=N[g].replace('\n',' '),color=COL[g]);axs[0].plot(x,curve[g]['safety baseline']['mean_progress'],ls='--',color=COL[g],alpha=.55);axs[1].plot(x,np.asarray(curve[g]['S500']['four_same_fraction'] if 'four_same_fraction' in curve[g]['S500'] else curve[g]['S500']['consistent_fraction'])*100,color=COL[g]);axs[1].plot(x,np.asarray(curve[g]['safety baseline']['consistent_fraction'])*100,ls='--',color=COL[g],alpha=.55);axs[2].plot(x*.05,np.asarray(curve[g]['S500']['success_cumulative_fraction'])*100,color=COL[g],label=N[g].replace('\n',' '))
 ex=curve['expert']['expert'];axs[2].plot(np.asarray(ex['steps'])*.05,np.asarray(ex['success_cumulative_fraction'])*100,color='black',ls=':',label='Centralized expert (privileged)');axs[0].set(xlabel='Recorded step',ylabel='Aggregate distance reduced (m)',title='More early progress with correction');axs[1].set(xlabel='Recorded step',ylabel='Team circulation consistent (%)',title='Early direction is not more consistent');axs[2].set(xlabel='Time (s)',ylabel='All-episode cumulative success (%)',title='Known failure remains in denominator');axs[0].legend(fontsize=6);axs[2].legend(fontsize=6,loc='lower right');fig.suptitle('Analysis 4: solid = correction13 to10 at500; dashed = safety layer only',y=1.03);fig.tight_layout();save(fig,'analysis4_correction_progress_arrival')
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for g in ['Z','V0','V1','V3','TD']:
  q=curve[g]['S500'];axs[0].plot(q['steps'],q['mean_progress'],label=N[g].replace('\n',' '),color=COL[g]);axs[1].plot(q['steps'],np.asarray(q['consistent_fraction'])*100,label=N[g].replace('\n',' '),color=COL[g])
 axs[0].set(xlabel='Recorded step',ylabel='Aggregate distance reduced (m)',title='Zero action loses sustained goal progress');axs[1].set(xlabel='Recorded step',ylabel='Team circulation consistent (%)',title='Choosing a common direction is insufficient');axs[0].legend(fontsize=7);fig.suptitle('Analysis 5: all traces use correction13 to10 at500',y=1.03);fig.tight_layout();save(fig,'analysis5_zero_controller')
 fig,axs=plt.subplots(1,3,figsize=(13,4.2));x=np.arange(3);gs=['V0','V1','V3'];axs[0].bar(x,[point(a2['groups'][g]['mixed_overall'])*100 for g in gs],color=[COL[g] for g in gs]);axs[0].set_xticks(x,[N[g] for g in gs]);axs[0].set(ylabel='Mixed direction (%)',title='Small mixed-direction change; CI crosses zero');x=np.arange(3);bottom=np.zeros(3)
 for key,label,color in [('success','Success','#009e73'),('collision','Collision','#d55e00'),('timeout','Timeout','#56b4e9')]:
  val=np.array([point(p[g]['nominal only'][key]['lower'])*100 for g in gs]);axs[1].bar(x,val,bottom=bottom,label=label,color=color);bottom+=val
 axs[1].set_xticks(x,[N[g] for g in gs]);axs[1].set(ylabel='Recorded episodes (%)',title='Fewer collisions, more timeouts');axs[1].legend(fontsize=7)
 q=a6['paired']['V3_minus_V1'];ks=['moving_no_progress_robot_seconds_difference','moving_no_progress_difference_both_timeout'];val=[point(q[k]) for k in ks];ci=np.asarray([q[k]['CI95'] for k in ks]);axs[2].errorbar([0,1],val,yerr=np.stack((np.asarray(val)-ci[:,0],ci[:,1]-np.asarray(val))),fmt='o',color=COL['V3'],capsize=5);axs[2].axhline(0,color='gray',ls='--');axs[2].set_xticks([0,1],['All paired episodes','Both episodes timeout']);axs[2].set(ylabel='Shared-noise minus relabeled (robot-seconds)',title='Observed duration matters');fig.tight_layout();save(fig,'analysis6_shared_noise_tradeoff')
 gs=['expert','TC','TD','V0','V1','V3','Z'];methods=['nominal only','safety baseline','F10','F13','S300','S500','RHR-fair','critic','oracle'];matrix=np.full((7,9),np.nan)
 for i,g in enumerate(gs):
  for j,a in enumerate(methods):
   q=p[g].get('expert' if g=='expert' and a=='nominal only' else a,{})
   if 'success' in q:matrix[i,j]=q['success']['lower']['estimate']*100
 fig,ax=plt.subplots(figsize=(13,5.3));cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#dddddd');im=ax.imshow(matrix,vmin=0,vmax=100,cmap=cmap,aspect='auto');ax.set_yticks(range(7),[N[g] for g in gs]);ax.set_xticks(range(9),[A[a] for a in methods],rotation=25,ha='right')
 for i in range(7):
  for j in range(9):ax.text(j,i,'missing' if not np.isfinite(matrix[i,j]) else f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if np.isfinite(matrix[i,j]) and matrix[i,j]>70 else 'black')
 fig.colorbar(im,ax=ax,label='Success lower bound (%)');ax.set_title('Analysis 7: all DEVELOPMENT; see PAPER_TABLE.csv for provenance, n and uncertainty');fig.tight_layout();save(fig,'analysis7_paper_heatmap')
 # Prespecified switch-window diagnostic, with survivor conditioning explicit.
 fig,ax=plt.subplots(figsize=(8,4));gs=['TD','V0','V1','V3','Z'];rr=[a4['switch'][g]['observed_600_post_minus_pre'] for g in gs];v=np.array([point(r) for r in rr]);ci=np.array([r['CI95'] for r in rr]);ax.errorbar(np.arange(5),v,yerr=np.stack((v-ci[:,0],ci[:,1]-v)),fmt='o',capsize=4,color='#0072b2');ax.axhline(0,color='gray',ls='--');ax.set_xticks(np.arange(5),[N[g] for g in gs]);ax.set(ylabel='Post500 minus pre500 progress rate (m/s)',title='Only episodes actually observed through600; survivor selection');fig.tight_layout();save(fig,'analysis4_switch_window')
 (H/'PLOT_PRECISION.json').write_text(json.dumps(precision,indent=2));print('figures complete',len(list(OUT.glob('*.png'))),flush=True)
if __name__=='__main__':main()
