from common import *
from horizon_statistics import summarize,ci
from audit_utils import source_inventory
OLD=D/'axis4a_addendum_v1'
def main():
    prereg();assert (HERE/'PSTAR_DECISION.receipt.json').exists();assert read(HERE/'SELECTED_PROGRESS_TEST.json')['passed'];assert not (HERE/'B_COMPLETE.receipt.json').exists();names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];parts={};differences=[]
    for i,H in enumerate([500,700,1000]):
        path=OLD/f'B_H{H}_LABELS.npz';z=np.load(path);assert z['controller_names'].tolist()==names;arms=z['arms'].tolist();lo=z['lower'];hi=z['upper'];events={k:z[k] for k in ['collision','timeout','unknown']};rel='S500' if H==700 else 'Srel';absolute='S500' if H==700 else 'F13' if H==500 else 'Sabs';choices={'nominal only':arms.index('nominal_only'),'safety baseline':arms.index('safety_baseline'),'F10':arms.index('F10'),'F13':arms.index('F13'),'Srel':arms.index(rel),'Sabs':arms.index(absolute),'Sprog':arms.index('Sprog')}
        if H==700:choices['S500']=arms.index('S500')
        for name,table in read(HERE/'FROZEN_REFERENCE_CHOICES.json')['controllers'].items():choices[name]=[arms.index(rel if table[n]=='S500' else table[n]) for n in names]
        q=summarize(lo,hi,events,arms,choices,202610163720+i);gain=q['paired']['A3 minus safety baseline']['CI_pp'];reg=q['methods']['A3']['mean_regret_pp'];q.update(H=H,method_applicability=dict(status='SUPPORTED' if gain[0]>0 and reg[1]<=3 else 'REJECTED' if gain[1]<=0 or reg[0]>3 else 'INCONCLUSIVE',gain_CI_pp=gain,full_menu_regret_pp=reg),source_file=str(path),source_sha256=sha(path),evidence='Inherited CONFIRMATORY（定义修订） labels; current offline reanalysis preserves source scope');parts[str(H)]=q;pr=arms.index('Sprog');re=arms.index(rel);differences.append((lo[:,:,pr]-hi[:,:,re],hi[:,:,pr]-lo[:,:,re]))
    rng=np.random.default_rng(202610163723);pooled=[];per={str(H):[] for H in [500,700,1000]}
    for _ in range(5000):
        cc=rng.integers(14,size=14);ss=rng.integers(32,size=(14,32));v=[]
        for H,(a,b) in zip([500,700,1000],differences):x=[a[cc[:,None],ss].mean(),b[cc[:,None],ss].mean()];per[str(H)].append(x);v.append(x)
        pooled.append(np.mean(v,axis=0))
    interval=ci(pooled);comp=dict(status='BETTER' if interval[0]>0 else 'WORSE' if interval[1]<0 else 'UNDETECTED',estimate_pp=(np.mean([[a.mean(),b.mean()] for a,b in differences],axis=0)*100).tolist(),CI_pp=interval,perH={str(H):dict(CI_pp=ci(per[str(H)]),estimate_pp=[a.mean()*100,b.mean()*100]) for H,(a,b) in zip([500,700,1000],differences)})
    out=dict(utc=utc(),status='COMPLETE',new_progress_rollouts=0,reused_progress_rollouts=2688,pstar=read(HERE/'PSTAR_DECISION.receipt.json')['pstar'],progress_minus_relative=comp,horizons=parts,source_old_decision=read(OLD/'PSTAR_DECISION.receipt.json'),definition='Only observed500699; fastest195 earlysuccess and2 earlyunknown excluded; lower threshold means slightly earlier switch than terminal-hold')
    write(HERE/'B_RESULTS.json',out);(HERE/'B_RESULTS.md').write_text('按进度切换时间表已由兼容历史实验执行，本轮独立重算门槛后复用全部2688条；未新跑同身份rollout。\n\n门槛为0.9681620884830543，仅699条实际500步读数；排除195条提前成功与2条提前数值失败，比终止保持定义略低、切换略早。\n\n'+json.dumps(comp,indent=2,ensure_ascii=False)+'\n');source_inventory('imports_analyze_B.json');seal('B_COMPLETE.receipt.json',[HERE/'B_RESULTS.json',HERE/'B_RESULTS.md',HERE/'PSTAR_DECISION.receipt.json',HERE/'SELECTED_PROGRESS_TEST.json'],new_progress_rollouts=0,compatible_labels_reused=2688)
if __name__=='__main__':main()
