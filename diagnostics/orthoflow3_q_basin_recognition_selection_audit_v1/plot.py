"""Static scientific figures; every representative is selected independently of scores."""
import csv, gzip, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent; FIG=HERE/'figures'
COLORS=['#edb94b','#d7dce2','#14967f']; BC=ListedColormap(COLORS); BN=BoundaryNorm([-1.5,-.5,.5,1.5],3)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def read(p):return json.loads(Path(p).read_text())
def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-700,700)))
def member(d):return np.where(d['success']>=15,1,np.where(d['failure']>=2,0,-1))
def load(name):return dict(np.load(HERE/'inputs'/f'{name}.npz'))
def short(name):
    return name.replace('matched_toy_give_way_','Toy ').replace('matched_ring_exchange_','Ring ').replace('ring_k16_','Ring K16 / Flow ').replace('ring_k2_','Ring K2 / Flow ').replace('ring_v11_t0','Ring K16 / Flow v11, true t0').replace('ring_v11_mid','Ring / Flow v11, midtrajectory').replace('toy_generator_rep48','Toy independent generator / 48 states').replace('db_transfer24','DB transfer / 24 states')
def arm(name):
    if name.startswith('matched'):return 'full_context',[17,23,41],'eta_only'
    if name.startswith('ring_v11'):return 'physical_context',[17,23,41],'eta_only'
    if name.startswith('ring_k16'):return 'controller_cv_full_context',[17,23,41],'controller_cv_eta_only'
    if name.startswith('ring_k2_8813'):
        ctl=int(name[-5:]);return ('trunk_only'if ctl<88136 else 'raw_bypass'),[17,23,41],'eta_only'
    if name=='db_transfer24':return 'full_context',['seed17','seed23','seed41'],'eta_only'
    return 'critic',['frozen'],'eta_only_kernel'
def key(kind,seed,condition='correct'):return 'z::'+f'{kind}__{seed}__{condition}'
def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)

def basin_map():
    names=['matched_toy_give_way_TT','matched_toy_give_way_FF','matched_ring_exchange_FF','ring_v11_t0','ring_k16_88139']
    selected=read(HERE/'figure_selection.json');metas={r['name']:r for r in read(HERE/'cohorts.json')}
    fig,axes=plt.subplots(len(names),4,figsize=(13,13),layout='constrained')
    for row,name in enumerate(names):
        d=load(name);i=selected[name];eta=d['eta'][i];b=member(d)[i];kind,seeds,_=arm(name);sizes=55+90*eta[:,2]/.75
        for col,ax in enumerate(axes[row]):
            if col==0:ax.scatter(eta[:,0],eta[:,1],c=b,cmap=BC,norm=BN,s=sizes,edgecolors='#35465a',linewidths=.7)
            else:
                z=d[key(kind,seeds[col-1])][i];im=ax.scatter(eta[:,0],eta[:,1],c=sigmoid(z),vmin=0,vmax=1,cmap='viridis',s=sizes,edgecolors='#35465a',linewidths=.7)
                j=int(np.argmax(z));ax.scatter(eta[j,0],eta[j,1],marker='*',s=255,c='none',edgecolors='#da3745',linewidths=1.7)
            for j,(x,y,_) in enumerate(eta):ax.annotate(str(j),(x,y),xytext=(4,4),textcoords='offset points',fontsize=6)
            ax.set_xlim(.45,1.3);ax.set_ylim(-.57,.57);ax.grid(alpha=.15);ax.set_xlabel(r'$\eta_1$');ax.set_ylabel(r'$\eta_2$')
            ax.set_title(('Empirical B15'if col==0 else f'Learned Q / seed {seeds[col-1]}')+'\n'+short(name))
        axes[row,0].text(.02,.02,f'state #{i}; {metas[name]["states"][i][-8:]}',transform=axes[row,0].transAxes,fontsize=7)
    fig.colorbar(im,ax=list(axes[:,1:].flat),fraction=.016,pad=.01,label='Predicted per-continuation success probability')
    fig.legend(handles=[Patch(color=COLORS[2],label='B15'),Patch(color=COLORS[1],label='non-B15'),Patch(color=COLORS[0],label='unresolved')],loc='outside lower center',ncol=3)
    fig.suptitle('A  |  Frozen Q over the empirical basin\nOutcome-blind hash representative; identical candidate points; marker area encodes eta3; red star = top-1',fontsize=13)
    save(fig,'figure_A_q_over_basin')

def atlas():
    with PdfPages(FIG/'all_states_atlas.pdf')as pdf:
        for meta in read(HERE/'cohorts.json'):
            name=meta['name'];d=load(name);kind,seeds,_=arm(name);b=member(d)
            ix=sorted(range(meta['N']),key=lambda i:hashlib.sha256(('q-basin-audit-v1|'+meta['states'][i]).encode()).hexdigest())
            fig,axes=plt.subplots(1,len(seeds)+1,figsize=(13,max(5,meta['N']*.16)),layout='constrained')
            for col,ax in enumerate(axes):
                if col==0:ax.imshow(b[ix],cmap=BC,norm=BN,aspect='auto',interpolation='nearest');title='B15 ground truth'
                else:
                    z=d[key(kind,seeds[col-1])][ix];im=ax.imshow(sigmoid(z),cmap='viridis',vmin=0,vmax=1,aspect='auto',interpolation='nearest');title=f'{kind}, seed {seeds[col-1]}'
                    ax.scatter(np.argmax(z,1),np.arange(len(ix)),s=12,marker='*',color='#f54454')
                ax.set_title(title,fontsize=8);ax.set_xlabel('Original candidate index');ax.set_xticks(range(meta['K']),labels=range(meta['K']),fontsize=6)
                ax.set_yticks(range(len(ix)),labels=[f'{i}: {meta["states"][i][-6:]}'for i in ix],fontsize=5)
            fig.colorbar(im,ax=list(axes[1:]),fraction=.018,label='Q score (trial success)')
            fig.suptitle(short(name)+' | all states, fixed hash order\n'+('Exact shared eta bank'if meta['shared_eta']else'Candidate index local to each state; no cross-state eta pairing'))
            pdf.savefig(fig,bbox_inches='tight');plt.close(fig)

def reversal():
    # Truth-defined events; no learned score or correctness enters example choice.
    examples=[]
    for name in ['matched_toy_give_way_TT','ring_k16_88139']:
        d=load(name);cs=np.load(HERE/'reversals'/f'{name}.npz')['cases::B15_swap']
        ev=min(cs.tolist(),key=lambda x:hashlib.sha256((name+'|'+str(x)).encode()).hexdigest());i,l,a,b,sg=ev
        kind,seeds,_=arm(name)
        examples.append(dict(label=short(name)+' / true B15 state reversal',coords=[f'state {i}',f'state {l}'],s=d['success'][[i,l]][:,[a,b]],f=d['failure'][[i,l]][:,[a,b]],
                             z=[d[key(kind,seed)][[i,l]][:,[a,b]]for seed in seeds],etas=[str(a),str(b)],rule=ev,cohort=name))
    rows=read(HERE/'inputs/current_source_field_rows.json');cs=next(r for r in read(HERE/'current_full_field_cases.json')if r['scene']=='ring_exchange')['reversals']['B15_swap']
    ev=min(cs,key=lambda x:hashlib.sha256(('field|'+str(x)).encode()).hexdigest());a,b,c,e,sg=ev;ix=np.array([[a,b],[c,e]]);z=dict(np.load(HERE/'inputs/current_source_fields.npz'))
    examples.append(dict(label='Ring / physical field reversal (source VAL; descriptive)',coords=['controller A','controller B'],s=np.array([r['s16']for r in rows])[ix],
                         f=np.array([r['f16']for r in rows])[ix],z=[z[f'full_context__seed{seed}__correct'][ix]for seed in [17,23,41]],etas=[rows[a]['eta_uid'][-6:],rows[b]['eta_uid'][-6:]],rule=ev,cohort='current_source_field_Ring'))
    fig,axes=plt.subplots(len(examples),4,figsize=(13,8.7),layout='constrained')
    for r,ex in enumerate(examples):
        for col,ax in enumerate(axes[r]):
            v=ex['s']/16 if col==0 else sigmoid(ex['z'][col-1]);colors=['#277ac1','#d97826']
            for j in range(2):
                ax.plot([0,1],v[:,j],'-o',c=colors[j],label='eta '+ex['etas'][j],lw=2)
                if col==0:ax.vlines([0,1],ex['s'][:,j]/16,(16-ex['f'][:,j])/16,color=colors[j],lw=3,alpha=.5)
            if col==0:ax.axhline(15/16,ls=':',color='#555',lw=.7);ax.legend(fontsize=7)
            ax.set_xticks([0,1],ex['coords']);ax.set_xlim(-.12,1.12);ax.set_ylim(-.04,1.06);ax.set_ylabel('Empirical Q16 bounds'if col==0 else'Predicted trial success')
            ax.set_title('Ground truth'if col==0 else f'Learned Q, seed {[17,23,41][col-1]}')
            if col:
                raw=ex['z'][col-1]
                ax.text(.5,.06,'Both rankings correct'if np.sign(raw[0,0]-raw[0,1])==ex['rule'][4] and np.sign(raw[1,0]-raw[1,1])==-ex['rule'][4]else'Joint reversal missed',transform=ax.transAxes,ha='center',fontsize=8)
        axes[r,0].text(-.03,1.19,ex['label'],transform=axes[r,0].transAxes,fontweight='bold',fontsize=9)
    fig.suptitle('B  |  Does Q follow a true ranking reversal?\nFirst truth-event by fixed hash; examples never selected by model success',fontsize=13)
    save(fig,'figure_B_ranking_reversals');(HERE/'reversal_figure_selection.json').write_text(json.dumps([{k:v for k,v in e.items()if k in ['label','coords','etas','rule','cohort']}for e in examples],indent=2)+'\n')

def decomposition():
    names=['matched_toy_give_way_TT','matched_ring_exchange_FF','ring_v11_t0','ring_k16_88139','toy_generator_rep48','ring_k2_88137']
    fig,axes=plt.subplots(3,2,figsize=(14,9),layout='constrained');cm=ListedColormap(['#b7bfc8','#ce5560','#13977f','#edb94b'])
    for ax,name in zip(axes.flat,names):
        d=load(name);b=member(d);kind,seeds,baseline=arm(name);eligible=(b==1).any(1);labels=['Oracle@K'];arr=[np.where(eligible,2,np.where((b==0).all(1),0,3))]
        cond='none'if name=='ring_v11_t0'else'correct'
        keys=[(baseline,seeds[0],cond)]+[(kind,s,'correct')for s in seeds]
        if name in ['ring_v11_t0','ring_k16_88139']:keys +=[(kind,23,'wrong_base'if name=='ring_v11_t0'else'wrong_controller')]
        for a,seed,co in keys:
            z=d[key(a,seed,co)];pick=np.argmax(z,1);chosen=b[np.arange(len(b)),pick];v=np.where(chosen==1,2,np.where(chosen<0,3,np.where(eligible,1,0)))
            arr.append(v);labels.append(f'{a} / {seed}'+(' / wrong C'if co.startswith('wrong')else''))
        ax.imshow(np.array(arr),cmap=cm,vmin=0,vmax=3,aspect='auto',interpolation='nearest');ax.set_yticks(range(len(labels)),labels,fontsize=7)
        ax.set_xticks(np.arange(0,len(b),max(1,len(b)//8)));ax.set_xlabel('Original held-out state index');ax.set_title(short(name),fontsize=10)
    fig.legend(handles=[Patch(color=c,label=l)for c,l in zip(cm.colors,['No robust candidate','Critic ranking error','Robust selection / oracle exists','Unresolved'])],loc='outside lower center',ncol=4)
    fig.suptitle('C  |  Selection decomposed state by state\nSame candidate set in every row; field generalization errors overlap ranking errors',fontsize=13)
    save(fig,'figure_C_selection_decomposition')

def main():
    FIG.mkdir(exist_ok=True);basin_map();atlas();reversal();decomposition();print('Figures A/B/C and complete 16-cohort atlas generated.')

if __name__=='__main__':main()
