"""Strengthen B3 with historical context-specific candidates, no test selection."""
import json
from scipy.stats import qmc
from runner import OUT,OLD,inputs,historical,dump,digest,make_case
from discovery_design import freeze,case,original_b3,merit
from screen import pool_plan


def main():
    p=freeze();old=[r for r in historical()if original_b3(r)and r['seed']==0 and 8<=r['state_index']<=15]
    dev=['xsc_hs_plus_w025','xsc_hs_box_n09_r014_f25','xsc_hs_swap_n07','hs_obst_n08_R140_r045'];mapping=dict(zip(p['test_scenes'],dev));ctls=inputs()[1];pools={}
    sobol=qmc.scale(qmc.Sobol(3,scramble=True,seed=2026102816).random_base2(3),[.5,-.5,0.],[1.25,.5,.75])
    for scene in p['test_scenes']:
        for ctl in ctls:
            candidates=[]
            for si in range(8,16):
                rr=[r for r in old if r['scene']==mapping[scene]and r['controller']==ctl['type_code']and r['state_index']==si]
                best=min(rr,key=merit);candidates.append(dict(eta=best['spec']['eta'][:3],coefficients={}))
            candidates.extend(dict(eta=e.tolist(),coefficients={})for e in sobol)
            pools[scene+'__'+ctl['type_code']]=candidates
    dump(OUT/'INFORMED_B3_FROZEN.json',dict(pools=pools,rule='best original legal B3 from each prior state8..15 in matching scene family/nominal +8 Sobol; no new test label selects eta'))
    for prefix,states,seed in [('test',[18,19],3),('confirm',[20,21],4)]:
        todo=[case(dict(scene=scene,controller=ctl['type_code'],state_index=si),sp,seed)for scene in p['test_scenes']for ctl in ctls for si in states for sp in pools[scene+'__'+ctl['type_code']]]
        pool_plan(prefix+'_informed_b3',todo,None)
    restart=json.load(open(OUT/'RESTART_FROZEN.json'));failed=set(json.load(open(OUT/'RESTART_LABELS.json'))['failed_after_B3'])
    points=qmc.scale(qmc.Sobol(3,scramble=True,seed=2026102817).random_base2(5),[.5,-.5,0.],[1.25,.5,.75]);todo=[]
    for c in restart['cases']:
        if c['scene']['name']not in failed:continue
        ctl=next(x for x in ctls if x['type_code']==c['controller'])
        todo.extend(make_case(c['scene'],ctl,0,dict(eta=e.tolist(),coefficients={}),restart['seed'])for e in points)
    pool_plan('restart_b3_48',todo,None)

if __name__=='__main__':main()
