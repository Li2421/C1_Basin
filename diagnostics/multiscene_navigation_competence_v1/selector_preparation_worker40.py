"""Disjoint pilot choices and label-free inputs, before target analysis."""
from common import *

def worker(phase,index,count):
    assert phase in ['inputs','predictions','analysis_AF']
    if phase=='inputs':
        from selector_data40 import generated_inputs
        from fewshot_choices40 import freeze
        families=read(HERE/'AF_MULTIAGENT_CAPABILITY_GATES.json')['eligible_families']
        for family in families[index::count]:
            p=HERE/f'{family}_SELECTOR_INPUTS40_FROZEN.receipt.json'
            if not p.exists():generated_inputs(family)
            else:assert all(sha(path)==h for path,h in read(p)['files'].items())
            p=HERE/'selector_choices40'/f'{family}_fewshot.receipt.json'
            if not p.exists():freeze(family)
            else:assert all(sha(path)==h for path,h in read(p)['files'].items())
    elif phase=='predictions':
        assert index==0 and count==1
        from selector_plan40 import freeze_predictions
        freeze_predictions()
    else:
        assert index==0 and count==1
        from corrective_analysis40 import analyze_AF
        analyze_AF()

if __name__=='__main__':worker(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
