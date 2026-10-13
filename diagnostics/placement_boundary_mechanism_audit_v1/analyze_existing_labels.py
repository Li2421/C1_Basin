"""Read frozen 16-seed TT/FF labels without generating new rollouts."""
import collections
import json
import sqlite3
from pathlib import Path
HERE=Path(__file__).resolve().parent
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')

def main():
    states=json.loads((HERE/'closed_loop_protocol.json').read_text())['selected_states']
    con=sqlite3.connect(HARD/'rollout.sqlite')
    rows=[]
    for st in states:
        uid=st['uid'];counts=collections.defaultdict(lambda:collections.defaultdict(lambda:[0,0,0]))
        for chain,i,text in con.execute('SELECT chain,eta_index,result_json FROM rollout WHERE state_uid=? AND chain IN ("TT","FF")',(uid,)):
            r=json.loads(text);v=counts[chain][i];v[0]+=int(r['success']);v[1]+=int(r['error'] is not None);v[2]+=1
        def label(v):
            s,m,n=v
            if n!=16:raise RuntimeError('incomplete Q16 panel')
            return 'TRUE' if s>=15 else 'FALSE' if s+m<15 else 'UNKNOWN'
        tt=[label(counts['TT'][i]) for i in range(16)];ff=[label(counts['FF'][i]) for i in range(16)]
        rows.append(dict(scene=st['scenario'],uid=uid,TT=tt,FF=ff,
            TT_true=tt.count('TRUE'),FF_true=ff.count('TRUE'),known_flips=sum(a!=b for a,b in zip(tt,ff) if a!='UNKNOWN' and b!='UNKNOWN'),
            unresolved_pairs=sum(a=='UNKNOWN' or b=='UNKNOWN' for a,b in zip(tt,ff))))
    con.close()
    (HERE/'existing_B15_labels.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('scene','uid','TT_true','FF_true','known_flips','unresolved_pairs')} for r in rows],indent=2))

if __name__=='__main__':main()
