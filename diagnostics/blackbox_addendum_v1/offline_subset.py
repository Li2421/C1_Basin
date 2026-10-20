from common import *
from statistics_core import summarize
from audit_utils import source_inventory

def main():
    prereg();assert not (HERE/'BLACKBOX_SUBSET_RESULTS.json').exists();f=read(R33/'FROZEN_CHOICES.json');z=np.load(R33/'LABELS.npz');names=z['controller_names'].tolist();arms=z['arms'].tolist();selected=[n for n in names if n.startswith('G') or n.startswith('DM')];assert len(selected)==12
    out={}
    for j,(key,nn) in enumerate([('黑盒子集',selected),('高斯行为克隆策略',[n for n in selected if n.startswith('G')]),('分散式模型预测控制',[n for n in selected if n.startswith('DM')])]):
        ix=[names.index(n) for n in nn];choices={'nominal only':arms.index('nominal_only'),'safety baseline':arms.index('safety_baseline'),'F10':arms.index('F10'),'F13':arms.index('F13'),'S500':arms.index('S500')}
        for k,label in [('A3_full','A3 full'),('A0_full','A0'),('G_E','G_E'),('mixed_full','mixed full'),('A3_fixed2','A3 fixed2'),('A0_fixed2','A0 fixed2')]:choices[label]=np.array([arms.index(f['controllers'][n][k]) for n in nn])
        v=summarize(key,z['lower'][ix],z['upper'][ix],{k:z[k][ix] for k in ['collision','timeout','unknown']},choices,list(range(2,8)),[2,3,4],202610163700+j);v.update(evidence='DEVELOPMENT（事后子集）',subset_basis='用户于2026-10-16给出的明确黑盒定义',controller_names=nn);out[key]=v
    out['历史四类型合并引用']=read(R33/'2d_all_RESULTS.json');out['历史四类型合并引用']['scope']='原报告的描述性引用，本轮未重算非黑盒类型'
    write(HERE/'BLACKBOX_SUBSET_RESULTS.json',out)
    lines=['本表仅新计算明确黑盒子集；历史四类型合并列是原报告引用。全部新计算为 DEVELOPMENT（事后子集）。','','| 子集 | 方法可用性 | 跨类型关系前提 | 混合类型训练优势 | 看controller特征的选择模型成功率界 |','|---|---|---|---|---|']
    for name,v in out.items():
        lines.append(f"| {name} | {v['CT1']['status']} | {v['CT2']['status']}（双向偏好数量 {v['CT2']['prefers_F10_by_10pp']}/{v['CT2']['prefers_F13_by_10pp']}） | {v['CT3']['status']} | {v['methods']['A3 full']['success_pct']} |")
    (HERE/'BLACKBOX_SUBSET_TABLE.md').write_text('\n'.join(lines)+'\n');source_inventory('imports_offline_subset.json');seal('F_COMPLETE.receipt.json',[HERE/'BLACKBOX_SUBSET_RESULTS.json',HERE/'BLACKBOX_SUBSET_TABLE.md',HERE/'offline_subset.py'],new_rollouts=0,evidence='DEVELOPMENT post-hoc subset')
if __name__=='__main__':main()
