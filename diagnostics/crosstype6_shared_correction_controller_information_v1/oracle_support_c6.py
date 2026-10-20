"""Quantify sparse donor support separately from task-performance estimates."""
from common import *
from oracle_c6 import load

def main():
    if (HERE/'TYPE_INFORMATION_SUPPORT.receipt.json').exists():return
    d,pack=load();types=d['types'];names=d['names'];tt=sorted(set(types));counts=np.zeros((122,24,6),np.float64)
    for si,p in enumerate(pack):
        M=p['M'];ids=p['ids'];known=M*((p['U']-p['Y'])<1-1e-12)
        bytype={t:M[types==t].sum(0) for t in tt};knownbytype={t:known[types==t].sum(0) for t in tt}
        for ci,t in enumerate(types):
            keys=ids[ci,ids[ci]>=0];same=bytype[t][keys]-M[ci,keys];other=sum((bytype[u][keys]>0).astype(int) for u in tt if u!=t);sameknown=knownbytype[t][keys]-known[ci,keys];otherknown=sum((knownbytype[u][keys]>0).astype(int) for u in tt if u!=t);common=(same>0)&(other>0)
            counts[ci,si]=[len(keys),common.sum(),np.mean(same<=0),np.mean(other<=0),np.mean((sameknown>0)&(otherknown>0)),float(common.sum()<3)]
    groups=read(HERE/'GROUP_MEMBERSHIP.json')['groups'];results={}
    for group,members in groups.items():
        mask=np.isin(names,members);tset=sorted(set(types[mask]))
        if not tset:results[group]=dict(controllers=0,gap='empty subgroup');continue
        mean=np.mean([counts[mask&(types==t)].mean((0,1)) for t in tset],0);fraction=np.mean([np.mean(counts[mask&(types==t),:,1]/counts[mask&(types==t),:,0]) for t in tset]);results[group]=dict(controllers=int(mask.sum()),mean_target_unique_actions=float(mean[0]),mean_common_supported_actions=float(mean[1]),type_equal_common_supported_action_fraction=float(fraction),target_action_fraction_without_same_type_donors=float(mean[2]),target_action_fraction_without_other_type_donors=float(mean[3]),target_action_fraction_with_known_label_support_in_both_pools=float(mean[4]),controller_state_fraction_with_fewer_than3_common_actions=float(mean[5]),minimum_common_supported_actions=int(counts[mask,:,1].min()))
    write(HERE/'TYPE_INFORMATION_SUPPORT.json',dict(utc=utc(),groups=results,definition='Exact physical actions actually evaluated for each held-out controller/state; target controller excluded from same-type donors; other types counted separately; known-label support means at least one nonunknown seed; all averages type/controller/state equal',no_prediction_substitutes=True))
    seal('TYPE_INFORMATION_SUPPORT.receipt.json',[HERE/'TYPE_INFORMATION_SUPPORT.json',HERE/'oracle_support_c6.py',HERE/'GROUP_MEMBERSHIP.json',HERE/'NEW_EVAL_BUNDLE.receipt.json'],descriptive_support_only=True,no_selection_changes=True)

if __name__=='__main__':main()
