"""Post-label descriptive clarification; frozen endpoints remain unchanged."""
from support import *
G=guard()
def main():
 r=read(HERE/'results.json');z=np.load(HERE/'LABELS.npz');names=list(z['controller_names']);families=z['families'];labels=z['labels'];rates=(labels==1).mean((2,3));types=np.array([0,1,1,2,2,2]);choices=read(HERE/'NEW_INSTANCE_CHOICES.json')['controllers'];out={}
 for f in FAMILIES:
  ids=np.flatnonzero(families==f);best=ARMS.index(r['policies']['primary']['families'][f]['best_fixed_arm']);policies={}
  for method in ['primary','secondary']:
   rows=[]
   for i in ids:
    a=ARMS.index(choices[names[i]][method]);oracle=int(np.argmax(rates[i]));ties=np.flatnonzero(np.isclose(rates[i],rates[i].max(),atol=1e-12,rtol=0));loss=float(rates[i,best]-rates[i,a]);rows.append(dict(controller=str(names[i]),chosen_arm=ARMS[a],individual_best_arm=ARMS[oracle],type_correct_vs_individual_best=bool(types[a]==types[oracle]),type_correct_vs_any_individual_tie=bool(types[a] in types[ties]),family_best_arm_on_this_controller=ARMS[best],paired_controller_loss=loss,paired_controller_catastrophe=loss>.10+1e-12))
   policies[method]=dict(type_accuracy_vs_individual_best=float(np.mean([q['type_correct_vs_individual_best'] for q in rows])),type_accuracy_allow_individual_ties=float(np.mean([q['type_correct_vs_any_individual_tie'] for q in rows])),paired_controller_catastrophes=sum(q['paired_controller_catastrophe'] for q in rows),rows=rows)
  out[f]=dict(full_name=FULL[f],policies=policies)
 write(HERE/'SUPPLEMENTARY_DESCRIPTIVES.json',dict(utc=utc(),evidence_level='POST_LABEL_DESCRIPTIVE_CLARIFICATION',frozen_results_unchanged=True,reason='Preregistered type accuracy uses family-best fixed type; preregistered catastrophe compares with family-average success. Add individual-oracle type accuracy and same-controller family-best-arm loss to distinguish selection error from within-family difficulty.',unknown=int((labels<0).sum()),families=out,guard=guard()))
 seal('SUPPLEMENTARY_DESCRIPTIVES.receipt.json',[HERE/'SUPPLEMENTARY_DESCRIPTIVES.json',HERE/'supplementary_descriptives.py',HERE/'results.json',HERE/'LABELS.npz'],changes_to_endpoints_or_choices=False,guard=guard())
if __name__=='__main__':main()
