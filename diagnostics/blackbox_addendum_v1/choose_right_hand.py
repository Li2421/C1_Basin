from common import *
from audit_utils import source_inventory

def main():
    cases=read(HERE/'right_hand_tune_CASES.json');db={r['rollout_uid']:r for r in read(HERE/'right_hand_tune_DB_ROWS.json')};grid=read(D/'teacher_student_macflow_v1d/RHR_GRID.json');results={};selected={}
    for pool,nc in [('Flow',14),('Gaussian',6),('MPC',6)]:
        table=[]
        for k,rule in enumerate(grid):
            cs=[c for c in cases if c['pool']==pool and c['grid_index']==k];assert len(cs)==nc*16;rr=[db[c['rollout_uid']] for c in cs];s=sum(r['success'] and not r['numerical_failure'] for r in rr);u=sum(r['numerical_failure'] for r in rr);table.append(dict(grid_index=k,rule=rule,success=s,unknown=u,lower=s/(nc*16),upper=(s+u)/(nc*16)))
        choice=min(table,key=lambda q:(-q['lower'],q['rule']['speed'],q['rule']['steps'],['always','slow','progress'].index(q['rule']['mode'])));selected[pool]=dict(rule=choice['rule'],grid_index=choice['grid_index'],tuning_lower=choice['lower'],tuning_upper=choice['upper'],unknown_selection_ambiguous=any(q['upper']>choice['lower'] for q in table if q is not choice));results[pool]=table
    write(HERE/'RIGHT_HAND_GRID_RESULTS.json',dict(utc=utc(),pools=results));write(HERE/'RIGHT_HAND_SELECTED.json',dict(utc=utc(),**selected,new_evaluation_labels_opened=False));seal('RIGHT_HAND_CHOICES_FROZEN.receipt.json',[HERE/'RIGHT_HAND_GRID_RESULTS.json',HERE/'RIGHT_HAND_SELECTED.json',HERE/'TUNING_STATES.json',HERE/'continuing_rhr.py',HERE/'trigger.py'],before_any_new_right_hand_evaluation_labels=True);source_inventory('imports_choose_right_hand.json')
if __name__=='__main__':main()
