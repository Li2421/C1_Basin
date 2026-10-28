"""Freeze the supplied six follow-up menus before their outcome data exist."""
import guard
from guard import HERE
import json,hashlib

def main():
    read=lambda n:json.loads((HERE/n).read_text())
    base=read('FAMILIES_64.json')['families'][4]['candidate_pool'];menu=read('ONE_WAY_MENU.json')['ordered']
    names={'B':'通道单向放行·合成一列','C':'只让等待方减速·一列','D':'定时交替放行','E':'整队让行','F':'预约到达时间','G':'去等候点'}
    families=[]
    for scheme,name in names.items():
        candidates=[]
        for k,original in enumerate(base):
            settings=dict(scheme=scheme)
            if scheme in('B','C','E'):settings['c']=[1.,.75,.5][k%3]
            elif scheme=='D':settings['T']=[4.,8.,12.][k%3]
            elif scheme=='F':settings['v_p']=[.1,.2,.3][k%3]
            elif scheme=='G':settings['menu']=menu[k]
            candidates.append(dict(original,coordination=settings))
        families.append(dict(scheme=scheme,name=name,candidate_pool=candidates,coefficients=3 if scheme=='G'else 1))
    target=HERE/'FAMILIES_FOLLOWUP.json';content=json.dumps(dict(families=families,order=list(names),threshold=dict(numerator=9,denominator=10),budget=35000,seed_reference=2026102901),ensure_ascii=False,indent=2)+'\n'
    if target.exists():assert target.read_text()==content
    else:target.write_text(content)
    (HERE/'FOLLOWUP_DEFINITIONS.json').write_text(json.dumps(dict(source='inputs/prompt_bottleneck_one_way_admission_v1_followup_20261029.md',source_sha256=hashlib.sha256((HERE/'inputs/prompt_bottleneck_one_way_admission_v1_followup_20261029.md').read_bytes()).hexdigest(),candidate_menu_sha256=hashlib.sha256(content.encode()).hexdigest(),runtime_sha256=hashlib.sha256((HERE/'followup_runtime.py').read_bytes()).hexdigest(),interpretations={'reservation_expiry':'Planned leave time expires the reservation; missed entry expires after enter+2s and cannot rebook until next step. Encounter means current inside/request relation to the same passage.','waiting_grid':'Global 0.05m lattice; robots pick nearest point to their current location, tie by exit distance then coordinates.','head_on_role':'Only selection is filtered by heading cosine; original inactivity rule controls reassignment.','signal_request_distance':'Default admission request distance 0.8m.'},zero_new_rollouts_at_freeze=True),ensure_ascii=False,indent=2)+'\n')
    print('Followup six menus frozen, 64 candidates each')

if __name__=='__main__':main()
