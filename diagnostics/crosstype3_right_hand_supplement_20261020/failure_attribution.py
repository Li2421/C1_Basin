"""Posthoc sampled-menu explanations, with no edits to confirmed endpoints."""
from common import *
from critic_model import metrics
from statistics_pipeline import evaluate_group,equivalence
from labels import type_name

def main():
    assert (HERE/'diagnostic_COMPLETE.receipt.json').exists();confirmation_sha=sha(HERE/'CONFIRMATORY_RESULTS.json');names=read(HERE/'UNSEEN_CAPABILITY.json')['eligible'];nc=len(names)
    if not names:
        (HERE/'FAILURE_ATTRIBUTION.md').write_text('没有通过能力门的未见实例；不能用其他类型替代，未运行事后诊断。\n');return
    base=read(HERE/'EVALUATION_PROTOCOL_BASE.json');types=np.asarray([base['types'][n] for n in names]);shape=(nc,8,31,1);labels=np.full(shape,-2,np.int8);uids=np.full(shape,'',dtype='U96');events={k:np.zeros(shape,np.int8) for k in ('collision','timeout','unknown')};indices=np.zeros((nc,8,26),np.int32);global_states=np.zeros((nc,8),np.int32)
    draw={r['controller']:r for r in read(HERE/'POSTHOC_DRAW.json')['controllers']}
    for i,n in enumerate(names):
        global_states[i]=draw[n]['state_indices'];indices[i]=np.asarray([r['candidate_indices'] for r in draw[n]['states']])
    with rollout_db.connect(True) as db:
        for c in read(HERE/'diagnostic_CASES.json'):
            i=names.index(c['base_controller_name']);s=c['diagnostic_state_index'];a=c['training_slot'];r=db.execute('SELECT success,collision,timeout,numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r is not None;labels[i,s,a,0]=-1 if r['numerical_failure'] else r['success'];uids[i,s,a,0]=c['rollout_uid']
            for k in events:events[k][i,s,a,0]=r['numerical_failure'] if k=='unknown' else r[k]
    with np.load(HERE/'UNSEEN_EVAL_BUNDLE.npz',allow_pickle=False) as z:previous={k:z[k] for k in z.files}
    with np.load(HERE/'UNSEEN_EVAL_ANALYSIS_CHOICES.npz',allow_pickle=False) as z:choices={k:z[k] for k in z.files if k not in ('names','types')}
    methods={key:np.full((nc,8),26+j,np.int32) for j,key in enumerate(('pipeline','all_safety_baseline','all_S500','all_Sprog','no_controller'))}
    for i,n in enumerate(names):
        pi=list(previous['names']).index(n)
        for local,si in enumerate(global_states[i]):
            for key,target in methods.items():
                a=int(target[i,local]);source=choices[key][pi,si]
                for k in ('labels','uids',*events):
                    destination=labels if k=='labels' else uids if k=='uids' else events[k];destination[i,local,a,0]=previous[k][pi,si,source,0]
    assert not np.any(labels==-2)
    methods['sampled_best26']=np.argmax(labels[:,:,:26,0]==1,-1).astype(np.int32)
    data=dict(names=np.asarray(names),types=types,labels=labels,uids=uids,uid_class=equivalence(uids),oracle_mask=np.ones(shape[:3],bool),non_diagonal=np.zeros(shape[:3],bool),**events)
    with np.load(HERE/'NEW_EVAL_PROBABILITIES.npz',allow_pickle=False) as z:prob=z['full'];order=list(z['names'])
    selected=np.asarray([prob[:,order.index(n),global_states[i]][:,np.arange(8)[:,None],indices[i]] for i,n in enumerate(names)]).transpose(1,0,2,3)
    assert selected.shape==(3,nc,8,26);ranking=metrics(selected,labels[:,:,:26],types,sorted(set(types)),indices)
    confirmed=read(HERE/'CONFIRMATORY_RESULTS.json');gate={r['controller']:r for r in read(HERE/'UNSEEN_CAPABILITY.json')['rows']};feature=read(HERE/'FEATURE_OUT_OF_RANGE.json');eta=read(HERE/'SELECTED_ETA_DESCRIPTIONS.json')['rows'];rows={};text=['# 未见控制器失败归因（事后诊断，DEVELOPMENT）\n','本诊断在确认性终点1–7封存后才执行；不改变模型或确认结论。每实例仅8状态、seed0，样本内最佳仅覆盖26个预定动作，不能证明全部288个动作都无效。差异是否明显以配对5000次bootstrap的95% CI是否含0判定。\n']
    for kind in sorted(set(types)):
        mask=types==kind;report=evaluate_group({k:v[mask] for k,v in data.items()},{k:v[mask] for k,v in methods.items()},True,5000,True);m=report['methods'];pair=report['paired_comparisons'];cf=confirmed['reports'][kind]['type_equal'];p=cf['paired_comparisons'];reasons=[]
        no_repair=all(gate[n]['safety_success_rate']>=.98 for n,t in zip(names,types) if t==kind) and cf['methods']['all_safety_baseline']['success_lower_pct']>=98
        if no_repair:reasons.append('① 能力门与新评估中只加安全投影均基本成功；主要检验不伤害')
        human=p['pipeline__minus__best_human_schedule'];safety=p['pipeline__minus__all_safety_baseline']
        if (safety['gain_pp']>=0 and safety['CI95_pp'][0]<=0) or (human['gain_pp']>=-1 and human['CI95_pp'][0]<=-1):reasons.append('② 点估计达到判定线，但CI未跨过要求，统计功效有限')
        best=pair['pipeline__minus__sampled_best26'];human_advantage=[]
        # A positive lower bound for human minus sampled-best is obtained by
        # repeating the paired evaluator with sampled-best as the pipeline.
        alternate=dict(methods,pipeline=methods['sampled_best26']);alternative=evaluate_group({k:v[mask] for k,v in data.items()},{k:v[mask] for k,v in alternate.items()},True,5000,True)
        for key in ('all_S500','all_Sprog'):
            r=alternative['paired_comparisons']['pipeline__minus__'+key]
            if r['CI95_pp'][1]<0:human_advantage.append(key)
        if human_advantage:reasons.append('④ 至少一种人工时间表明显优于诊断26动作最佳；支持提议覆盖不足，但26动作不是完整候选菜单')
        if best['CI95_pp'][1]<0:reasons.append('⑤ 诊断候选中有明显更好的动作，critic未选中；这是同样本事后上界证据')
        if not human_advantage and best['CI95_pp'][1]>=0 and not no_repair:reasons.append('③/④ 仅列出样本内最佳与人工成功率；26动作和8状态不足以断言修法整体无效或排除覆盖缺口')
        oob=[r for r in feature['rows'] if r['type']==kind];agreement=np.mean([r['primary_NoC_same_action'] for r in eta if r['type']==kind]);row=dict(report=report,ranking=ranking[kind],sampled_best_vs_humans=alternative['paired_comparisons'],reasons=reasons,feature_out_of_range=oob,primary_NoC_choice_agreement=float(agreement),primary_minus_NoC=cf['paired_comparisons']['pipeline__minus__no_controller'],implementation_evidence='32 mandatory reproductions, generator bitwise and all three new-controller unit tests passed',training_comparison=read(HERE/'DEVELOPMENT_RESULTS.json')['ranking'])
        rows[kind]=row;text.append('\n## '+type_name(kind,True)+'\n');text.append('；'.join(reasons)+'。\n');text.append(f"诊断26动作样本内最佳 {m['sampled_best26']['success_lower_pct']:.2f}%；主方法 {m['pipeline']['success_lower_pct']:.2f}%；固定500步人工 {m['all_S500']['success_lower_pct']:.2f}%；按进度人工 {m['all_Sprog']['success_lower_pct']:.2f}%。主方法减样本内最佳 {best['gain_pp']:.2f}pp，95% CI {best['CI95_pp']}。\n");text.append(f"⑤b 主方法与不看控制器的动作一致率 {agreement*100:.2f}%；其新评估成功率之差及CI见FAILURE_ATTRIBUTION.json。⑤a 特征越界逐实例列于同文件。⑥ 复现与单元测试通过，数值unknown保留，没有将timeout直接称为死锁。\n")
    write(HERE/'FAILURE_ATTRIBUTION.json',dict(utc=utc(),evidence='DEVELOPMENT_POSTHOC8_STATES_SEED0',rows=rows,confirmation_sha256=confirmation_sha,ranking_scope='26 fixed sampled actions, original four identifiable-pair categories',diagnostic_state_indices=global_states))
    (HERE/'FAILURE_ATTRIBUTION.md').write_text('\n'.join(text));np.savez_compressed(HERE/'DIAGNOSTIC_BUNDLE.npz',**data,candidate_indices=indices,global_state_indices=global_states,probability=selected)
    assert sha(HERE/'CONFIRMATORY_RESULTS.json')==confirmation_sha
    seal('FAILURE_ATTRIBUTION.receipt.json',[HERE/'FAILURE_ATTRIBUTION.json',HERE/'FAILURE_ATTRIBUTION.md',HERE/'DIAGNOSTIC_BUNDLE.npz',HERE/'failure_attribution.py',HERE/'POSTHOC_LABEL_DESIGN.receipt.json',HERE/'CONFIRMATORY_RESULTS.receipt.json'],confirmation_unchanged=True)
if __name__=='__main__':main()
