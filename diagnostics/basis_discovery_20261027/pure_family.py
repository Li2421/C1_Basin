"""Evaluate the compressed family itself, not only its union with retained B3."""
import json
from runner import OUT,inputs,make_case
from discovery_design import case
from screen import pool_plan


def main():
    final=json.load(open(OUT/'CONFIRMATORY_FINAL.json'));ctls=inputs()[1];seed=final['seeds'][0]
    cohort=[dict(scene=s,controller=ctl['type_code'],state_index=si)for s in final['scenes']for ctl in ctls for si in final['states']]
    pool_plan('confirm_pure_family',[case(c,sp,seed)for c in cohort for sp in final['oracle_pool']],None)
    restart=json.load(open(OUT/'RESTART_FROZEN.json'));todo=[]
    for c in restart['cases']:
        ctl=next(x for x in ctls if x['type_code']==c['controller'])
        for sp in final['oracle_pool']:todo.append(make_case(c['scene'],ctl,0,sp,restart['seed']))
    pool_plan('restart_pure_family',todo,None)

if __name__=='__main__':main()
