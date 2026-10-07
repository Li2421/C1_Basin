#!/usr/bin/env python3
"""Outcome-blind 24-state Double-Bottleneck confirmation pool."""
from pathlib import Path
import json
from diagnostics.double_bottleneck_initial_state_coverage.tools.generate_pools import _generate_split
from double_bottleneck.expert_dataset import save_dataset

OUT=Path(__file__).resolve().parent/'fresh_double_pool'
def main():
 if OUT.exists():
  print(json.dumps({'reused':True,'path':str(OUT)}));return
 specs,records,rejections=_generate_split('ring_revision_fresh_test',8,4)
 manifest=save_dataset(OUT,records,generation_metadata={'schema':'ring_revision_double_fresh_test_v1','scientific_split':'fresh_test','storage_split_label':'val','accepted_states_per_regime':8,'modes_per_state':8,'adaptive_sampling':False,'generated_before_proposals':True,'generated_before_outcomes':True})
 (OUT/'initial_state_rejections.json').write_text(json.dumps(rejections,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'states':len(specs),'records':len(records),'rejections':len(rejections),'path':str(OUT)}))
if __name__=='__main__':main()
