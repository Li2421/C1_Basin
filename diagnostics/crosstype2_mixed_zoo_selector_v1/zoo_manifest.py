from common import *
from zoo_design import profiles,LOCKED
from baselines import LEARNED
from deliver_tables import TYPE_NAMES
import csv


def main():
    base=profiles();cap={}
    for cohort in ('DEVELOPMENT','PPO','LOCKED'):
        for row in read(HERE/f'{cohort}_CAPABILITY.json')['rows']:cap[row['controller']]=row
    configs={c['name']:c for c in read(HERE/'IPPO_TRAINING_CONFIGS.json')};old=D/'crosstype1_learned_zoo_selector_v1'
    with (old/'ZOO_MANIFEST.csv').open() as f:previous={r['controller']:r for r in csv.DictReader(f)}
    unsealed=read(HERE/'LOCKED_CAPABILITY_UNBLINDED_AFTER_CONFIRMATION.json')['controllers'] if (HERE/'LOCKED_CAPABILITY_UNBLINDED_AFTER_CONFIRMATION.json').exists() else {}
    rows=[]
    for name in base['controller_order']:
        kind=base['types'][name];payload=base['profiles'][name]['payload'];gate=cap.get(name);cfg=configs.get(name,{})
        row=dict(controller=name,type=TYPE_NAMES[kind],type_code=kind,family='学习型' if kind in LEARNED else '搜索型' if kind=='prioritized_astar' else '优化型',cohort='锁定' if kind in LOCKED else '开发',effective_execution_chain='FF' if kind=='Flow' else 'TT',legacy_metadata_chain_tag=payload.get('chain',''),recipe=cfg.get('recipe',payload.get('source_Flow_recipe',previous.get(name,{}).get('recipe','固定预注册网格'))),checkpoint=payload.get('flow_checkpoint_path') or '',checkpoint_sha256=payload.get('flow_checkpoint_sha256') or '',parameters=json.dumps(payload.get('nominal_parameters'),ensure_ascii=False),controller_uid=base['profiles'][name]['controller_uid'],capability_passed=gate['passed'] if gate else True,gate_evidence='本轮先导16状态四臂' if gate else '历史合格实例直接保留',no_repair_needed=unsealed[name]['no_repair_needed'] if name in unsealed else gate.get('no_repair_needed','冻结前盲态') if gate else '沿用历史证据，见前轮清单',training_robot_steps=cfg.get('robot_steps',0))
        rows.append(row)
    for i in range(6):rows.append(dict(controller=f'implicit{i}',type=TYPE_NAMES['implicit'],type_code='implicit',family='学习型',cohort='可选锁定缺口',effective_execution_chain='N/A',legacy_metadata_chain_tag='',recipe='原两峰合成门未通过；不替换',checkpoint='',checkpoint_sha256='',parameters='null',controller_uid='',capability_passed=False,gate_evidence='原合成门不通过，未进行Ring训练/能力门/评估',no_repair_needed='N/A',training_robot_steps=0))
    with (HERE/'ZOO_MANIFEST.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(HERE/'ZOO_COUNTS.json',dict(utc=utc(),instantiated=104,development=92,locked=12,optional_missing=6,development_types=14,locked_types_present=2,optional_locked_type_gap=True,rows=len(rows)))
    seal('ZOO_MANIFEST.receipt.json',[HERE/'ZOO_MANIFEST.csv',HERE/'ZOO_COUNTS.json',HERE/'zoo_manifest.py'])
    print('Zoo manifest written:',len(rows),'including six explicitly missing optional policies')


if __name__=='__main__':main()
