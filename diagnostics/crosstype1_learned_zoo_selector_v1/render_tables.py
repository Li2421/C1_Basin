"""Render every requested method/type, retaining unavailable cells explicitly."""
from common import *
import csv

TYPE_NAMES={'Flow':'环形交换场景流匹配策略','gaussian':'高斯行为克隆策略','deterministic':'确定性行为克隆策略',
    'mixture':'混合密度行为克隆策略','diffusion':'扩散行为克隆策略','chunking':'动作分块行为克隆策略',
    'recurrent':'循环网络行为克隆策略','cvae':'条件变分自编码器行为克隆策略',
    'ring_ippo':'参数共享独立近端策略优化强化学习策略','implicit':'隐式能量行为克隆策略'}
METHODS={'nominal_only':'只执行 controller 动作','all_safety_baseline':'只加安全投影',
    'all_F10':'全程 η10','all_F13':'全程 η13','all_S100':'前100步 η13后 η10','all_S300':'前300步 η13后 η10',
    'all_S500':'按比例切换时间表','all_Sprog':'按进度切换时间表','all_RHR':'持续右手规则',
    'pipeline':'本轮零样本选择器','frozen_round38':'原冻结扩展选择模型','r5_pca2_pipeline':'行为指纹同结构选择器',
    'A0':'原不看 controller 的 critic','A3':'原 controller 级 critic','G_E':'探测规则选择',
    'few_labels_2':'前2个调参状态选臂','few_labels_4':'前4个调参状态选臂','few_labels_8':'前8个调参状态选臂',
    'crossfit_oracle':'交叉拟合事后最佳','in_sample_oracle':'样本内事后最佳','type_best_fixed_oracle':'每类型最佳固定臂'}


def write_csv(path,rows,fields=None):
    with path.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)


def main(iteration):
    development=read(HERE/'iterations'/iteration/'RESULTS.json')
    locked=read(HERE/'CONFIRMATORY_RESULTS.json') if (HERE/'CONFIRMATORY_RESULTS.json').exists() else None
    groups=[]
    for t,name in TYPE_NAMES.items():
        src=locked if t in ('ring_ippo','implicit') else development
        record=src['summaries'].get(t,{}).get('type_equal') if src else None
        groups.append((t,name,'CONFIRMATORY' if t in ('ring_ippo','implicit') else 'DEVELOPMENT',record))
    groups.append(('development_combined','开发类型合并（留一类型，类型等权）','DEVELOPMENT',development['summaries']['combined']['type_equal']))
    groups.append(('locked_combined','锁定类型合并（类型等权）','CONFIRMATORY',locked['summaries']['combined']['type_equal'] if locked else None))
    wide=[];long=[];gaps=[]
    for code,name,level,record in groups:
        row=dict(type=code,type_name=name,evidence_level=level)
        for method in METHODS:
            item=record['methods'].get(method) if record else None
            if item is None:
                reason=(record.get('unavailable_methods',{}).get(method,'No complete eligible cohort evidence') if record else 'Type or phase unavailable')
                row[method]='N/A';gaps.append(dict(type=code,method=method,reason=reason))
                long.append(dict(type=code,type_name=name,method=method,method_name=METHODS[method],evidence_level=level,status='N/A',reason=reason))
            else:
                ci=item['success_CI95_pct'];row[method]=f"{item['success_lower_pct']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
                long.append(dict(type=code,type_name=name,method=method,method_name=METHODS[method],evidence_level=level,status='AVAILABLE',reason='',**item))
        wide.append(row)
    fields=list(wide[0]);write_csv(HERE/'AXIS_TABLE.csv',wide)
    longfields=list(dict.fromkeys(k for r in long for k in r));write_csv(HERE/'AXIS_METRICS.csv',long,longfields)
    md=['# 跨学习型类型结果表','',f'开发模型迭代：`{iteration}`。成功率为下界百分数，方括号为5000次配对成簇bootstrap的95%区间；unknown另报上界。',
        '类型等权先重抽类型，再重抽controller和state；seed重复保持成簇。Oracle在每次bootstrap中重选。',
        '锁定类型缺失时，锁定合并仅描述现有实例；不能把该行当作两类型完整主终点。','',
        '方法全称（代码只作列身份）：','']
    md.extend(f'- {full}（`{code}`）' for code,full in METHODS.items())
    md+=['','| 类型 | 证据等级 | '+' | '.join(METHODS)+' |','| --- | --- | '+' | '.join('---' for _ in METHODS)+' |']
    for r in wide:md.append('| '+r['type_name']+' | '+r['evidence_level']+' | '+' | '.join(r[m] for m in METHODS)+' |')
    md+=['','碰撞率、超时率、unknown率、成功率上下界、平均/最大遗憾、伤害数与灾难数及其区间见 `AXIS_METRICS.csv`；N/A原因见 `AXIS_TABLE_GAPS.json`。',
        'Controller等权合并、每实例选择与逐条转移见原统计JSON和明细CSV，不把timeout改称deadlock。','']
    (HERE/'AXIS_TABLE.md').write_text('\n'.join(md));write(HERE/'AXIS_TABLE_GAPS.json',gaps)
    details=read(HERE/'iterations'/iteration/'CONTROLLER_DETAILS.json')
    details=[dict(cohort='development',**r) for r in details]
    if locked:details += [dict(cohort='locked',**r) for r in read(HERE/'LOCKED_CONTROLLER_DETAILS.json')]
    detailfields=list(dict.fromkeys(k for r in details for k in r));write_csv(HERE/'CONTROLLER_SELECTION_DETAILS.csv',details,detailfields)
    summaries=[]
    for result,cohort in ((development,'development'),(locked,'locked')):
        if not result:continue
        for aggregation,record in result['summaries']['combined'].items():
            for method,item in record['methods'].items():summaries.append(dict(cohort=cohort,aggregation=aggregation,method=method,**item))
    write_csv(HERE/'COMBINED_AGGREGATIONS.csv',summaries)
    print('Axis and controller tables rendered; all unavailable methods and types retained')


if __name__=='__main__':main(sys.argv[1])
