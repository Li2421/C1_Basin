import bootstrap
from experiment import *
from audit_supplement import V2,H,evidence
from collections import defaultdict
import copy
import numpy as np
SCENES=('lit_narrow_passage','lit_crowded_random','lit_circle_five')

def schedule(a,b,theta):return [np.asarray(a,np.float64).tobytes().hex(),np.asarray(b,np.float64).tobytes().hex(),theta]
def main():
    import numpy as np
    index={};templates={};selection={};sources=[]
    for scene in SCENES:
        split_path=V2/f'LEARNING_SPLIT_{scene}.json';evidence(split_path);split=read(split_path)
        qname='capability_v2_lit_narrow_passage' if scene=='lit_narrow_passage' else 'capability_'+scene
        qp=V2/f'{qname}_QUALIFICATION.json';evidence(qp);q={r['controller'] for r in read(qp) if r['qualified']}
        train=split['training_controllers'];selected={t:next((c for c in train if c['type_code']==t and c['controller'] in q),None) for t in sorted({c['type_code'] for c in train})}
        selection[scene]=dict(training_states=split['training'][:16],evaluation_states=split['evaluation'],training_controllers=selected,qualified=sorted(q),missing_types=[t for t,c in selected.items() if c is None])
        keep={(s['N'],s['index']) for s in split['training'][:16]+split['evaluation']}
        for stage in (f'distributed_baseline_{scene}',f'distributed_revised_fixed_{scene}',f'learning_labels_revised_20261021_{scene}'):
            p=V2/f'{stage}_CASES.json';h=evidence(p);count=0
            for c in read(p):
                s=c['state'];ctl=c['controller_payload']['nominal']['controller'];key=canonical([scene,ctl,s['N'],s['index']])
                if (s['N'],s['index']) not in keep:continue
                if key not in templates:templates[key]=c
                if c['seed']!=0:continue
                sem=canonical([scene,ctl,s['N'],s['index'],0,schedule(c['eta_a'],c['eta_b'],c['theta'])]);path=V2/stage/'raw'/f"{c['rollout_uid']}.json"
                if not path.exists():continue
                if sem not in index:index[sem]=dict(rollout_uid=c['rollout_uid'],path=str(path),case_manifest=str(p))
                count+=1
            sources.append(dict(path=str(p),sha256=h,compatible_rows=count))
    write(HERE/'HISTORY_SELECTION.json',selection);write(HERE/'HISTORY_EXECUTION_INDEX.json',index);write(HERE/'HISTORY_TEMPLATES.json',templates);write(HERE/'HISTORY_SOURCES.json',sources)
    text=['# 历史证据检查','',f'第一部分已完成，COMPLETE SHA256={evidence(H/"COMPLETE.json")}，0新研究重跑；其双积分器轨迹不能回答本轮SI η筛选。',
      '', 'v2单积分器分布式baseline、修订fixed、修订15动作training labels按完整物理state/controller/ηfloat64/schedule/seed匹配。方法名和critic编码不改变物理执行语义；已有unknown保留并作为失败下界，不能重试。旧controller参数/代码、state拆分、case fingerprint保留。',
      '', 'V2的旧原修正语义非零η与本轮contact-gated执行不兼容；只加安全投影可兼容。集中式/双积分器不同不复用。新swap8几何/局部感知/移动障碍不存在旧证据，必须新能力门与rollout，不训练新controller或critic。',
      '',f'已发现{len(index)}个兼容历史语义键（只索引本轮可能用到的训练/评估state）；实际复用须官方DB preflight精确UID及raw/trace SHA核验。HISTORY_SOURCES列9份case清单及SHA，原raw在首次使用时另记SHA。']
    for s,r in selection.items():text.append(f"\n{s}：前16训练state；评估{len(r['evaluation_states'])}state，选中{sum(c is not None for c in r['training_controllers'].values())}/8训练类型。缺类型{r['missing_types']}。")
    text.append('\n圆形MPPI的v2训练实例0/1均不合格，严格照原训练controller范围无法提供第8类；不从评估controller补入，7类筛选为明确协议缺口，不冒充要求的8类完整比较。其余独立工作继续。')
    (HERE/'HISTORY_CHECK.md').write_text('\n'.join(text)+'\n')
    print('Historical semantic entries',len(index),'templates',len(templates),flush=True)
if __name__=='__main__':
    import numpy as np
    main()
