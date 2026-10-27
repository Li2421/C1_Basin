import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from analyze import load, splits, own_honest, cross_loss
M=load(sys.argv[1:]); S=splits(M); own={n:own_honest(M,S,n) for n in M}
for n in sorted(M):
    if 'box_n' in n or n=='lit_circle_five':
        print(f"{n:24s} circle-five's eta on it: {100*cross_loss(M,S,own,'lit_circle_five',n):+6.1f} | its eta on circle-five: {100*cross_loss(M,S,own,n,'lit_circle_five'):+6.1f} | its eta on swap8-obstacles: {100*cross_loss(M,S,own,n,'lit_swap8_obstacles'):+6.1f} | swap6's eta on it: {100*cross_loss(M,S,own,'xsc_swap_n6',n):+6.1f}")
