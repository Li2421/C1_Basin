from experiment import *
from registry import case,register

def main():
    prereg();states=read(HERE/'PAPER_STATES.json');rows=[]
    hashes={n:sha(HERE/n) for n in ('paper.py','states.py','PREREGISTRATION_ADDENDUM_01.md')}
    external=read(HERE/'EXTERNAL_SOURCES.json')
    for s in states:
        scene=s['scene']
        methods=['impc_dr','buffered_voronoi_right'] if scene in ('lit_narrow_passage','lit_crowded_random') else ['circle_centralized','circle_df','circle_dr'] if scene=='lit_circle_five' else ['doorway_off','doorway_on']
        for method in methods:
            physical=dict(scene=scene,N=s['N'],version='paper',dt=.2 if scene=='lit_narrow_passage' else .15 if scene=='lit_crowded_random' else .05 if scene=='lit_circle_five' else .1,radius=2. if scene=='lit_circle_five' else .1 if scene=='lit_doorway' else .15,source_hashes=hashes)
            rows.append(case(s,'paper',method,physical,dict(source_hashes=hashes,official_commits={k:v['commit'] for k,v in external.items()},method=method,preregistration=sha(HERE/'PREREGISTRATION.md'))))
    register('paper_primary',rows)

if __name__=='__main__':main()
