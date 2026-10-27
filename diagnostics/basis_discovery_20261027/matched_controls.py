"""Direct P/P2 and R/R2/R3 comparisons, explicitly distinguishing RMS scaling."""
import json
from runner import OUT,dump
from discovery_design import freeze,case
from screen import pool_plan
from shared_control.basis_families import ORTHOFLOW3_SCALE


def main():
    tasks=[];labels={}
    for c in freeze()['cases']:
        for sign in(-.5,.5):
            specs={'P_original':dict(eta=[.875,sign,.375],coefficients={}),
                   'P2_native':dict(eta=[.875,0.,.375],coefficients={'P2':sign}),
                   'P2_matched_RMS':dict(eta=[.875,0.,.375],coefficients={'P2':sign*ORTHOFLOW3_SCALE})}
            for label,sp in specs.items():
                cc=case(c,sp);tasks.append(cc);labels.setdefault(label,[]).append(cc['rollout_uid'])
        for gain in(.375,.75):
            for name in ('R','R2','R3'):
                sp=dict(eta=[.875,0.,gain if name=='R'else 0.],coefficients={}if name=='R'else{name:gain})
                cc=case(c,sp);tasks.append(cc);labels.setdefault(name,[]).append(cc['rollout_uid'])
    dump(OUT/'MATCHED_CONTROLS.json',dict(labels=labels,reason='P2_matched_RMS exceeds default new-column box to match original P scaling; reported separately.'))
    pool_plan('matched',tasks,None)

if __name__=='__main__':main()
