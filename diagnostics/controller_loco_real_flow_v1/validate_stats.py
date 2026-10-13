"""Small adversarial checks for paired numerical-unknown contrasts."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from diagnostics.controller_loco_real_flow_v1.analyze import picked

def main():
    # A selector and baseline choosing the same unknown outcome have zero
    # paired difference, rather than a fictitious negative unknown penalty.
    Y=np.zeros((1,16));U=np.zeros((1,16));U[0,13]=1
    choice=np.array([[13]]);j=13
    selected=picked(Y[None],choice)
    conservative=(selected-Y[:,j][None]-U[:,j][None]*(choice!=j))
    assert conservative.item()==0
    # Distinct η: comparator unknown receives success in the worst-case bound.
    Y[0,4]=1;choice=np.array([[4]])
    conservative=(picked(Y[None],choice)-Y[:,j][None]-U[:,j][None]*(choice!=j))
    assert conservative.item()==0
    # Matched/mismatched use each exact same-state, same-seed η label.
    wrong=np.array([[[13]]]);wy=picked(Y[None,None],wrong)
    wu=picked(U[None,None],wrong)
    assert (picked(Y[None],choice)[:,None]-wy-wu*(choice[:,None]!=wrong)).item()==0
    print('PASS paired unknown cancellation and distinct-eta lower bound')

if __name__=='__main__':main()
