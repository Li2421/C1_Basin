from plan_D import *
from rhr_reuse import attach_references
def evaluation():
    assert (HERE/'RIGHT_HAND_CHOICES_FROZEN.receipt.json').exists();selected=read(HERE/'RIGHT_HAND_SELECTED.json');p=protocol();rows=[];names=source()['controller_names']
    for label,states,H,r,seeds in [('D1',read(R32/'EVAL_STATES.json')[:64],700,.9,[0,1]),('D2',read(R32/'EVAL_STATES.json')[64:],700,.9,[0,1]),('D3_H500',read(R35/'EVAL_STATES_ORIGINAL.json'),500,.9,[0,1]),('D3_H1000',read(R35/'EVAL_STATES_ORIGINAL.json'),1000,.9,[0,1])]+[(f'D4_r{radius}',read(HERE/'A2_ORIGINAL_STATES.json'),700,radius,[0]) for radius in [.8,.85,.95,1.]]:
        for j,old in enumerate(states):
            s=ring_state(p,old,H,r);s['study_index']=j
            for n in names:
                for seed in seeds:
                    c=ring_case(p,s,n,H,r,'RHR',seed,rule=selected['Flow']['rule']);c.update(engine='continuing_rhr',subunit=label,pool='Flow');rows.append(c)
    for old in read(R33/'EVAL_STATES.json'):
        s=novel_state(p,old)
        for n in [f'G{i}' for i in range(6)]+[f'DM{i}' for i in range(6)]:
            pool='Gaussian' if n.startswith('G') else 'MPC'
            for seed in [0,1]:
                c=novel_case(p,s,n,selected[pool]['rule'],seed);c.update(subunit='D5',pool=pool);rows.append(c)
    expected=6144;assert len(rows)==expected,(len(rows),expected);attach_references(p,rows);register('right_hand_evaluation',p,rows)
if __name__=='__main__':evaluation()
