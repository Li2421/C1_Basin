import json, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, KEYS, ETA
def gap_once(M, sub, rng, reps=40):
    v=[]
    for _ in range(reps):
        tr=[];sc=[];own=[]
        for n in sub:
            p=rng.permutation(len(M[n])); h=len(p)//2; a,b=M[n][p[:h]],M[n][p[h:]]
            tr.append(a.mean(0)); sc.append(b.mean(0)); s=a.mean(0); t=np.flatnonzero(s>=s.max()-1e-12); own.append(b[:,t].mean())
        g=int(np.argmax(np.mean(tr,0))); v.append(np.mean(own)-np.mean([s[g] for s in sc]))
    return float(np.mean(v))
def ci(M, sub, B=400, seed=2026102631):
    rng=np.random.default_rng(seed); point=gap_once(M,sub,np.random.default_rng(0),reps=200); boots=[]
    for _ in range(B):
        ss=[sub[i] for i in rng.integers(0,len(sub),len(sub))]
        MB={}; names=[]
        for j,n in enumerate(ss):
            k=f'{n}#{j}'; MB[k]=M[n][rng.integers(0,len(M[n]),len(M[n]))]; names.append(k)
        boots.append(gap_once(MB,names,rng,reps=20))
    return point, np.percentile(boots,[2.5,97.5]).tolist()
if __name__=='__main__':
    M=load(sys.argv[2:]); sub=sys.argv[1].split(',')
    p,(lo,hi)=ci(M,sub); print(f'{len(sub)} scenes: one global eta below own by {100*p:+.1f} pp, 95% interval [{100*lo:+.1f}, {100*hi:+.1f}]')

def ci_states(M, sub, B=400, seed=2026102632):
    rng=np.random.default_rng(seed); boots=[]
    for _ in range(B):
        MB={n:M[n][rng.integers(0,len(M[n]),len(M[n]))] for n in sub}
        boots.append(gap_once(MB,sub,rng,reps=20))
    return np.percentile(boots,[2.5,97.5]).tolist()
