from common import *
from design_generated import identities
from registry import register
cases=read(HERE/'expert_gate_v2_CASES.json');new=[];rows=[]
for case in cases:
 c,r=identities(case['state'],'navigation_field','navigation');assert c['state_uid']==case['state_uid'];new.append(c);rows+=r
register('expert_gate_v3',new,rows)
seal('EXPERT_GATE_IMPLEMENTATION_V3.receipt.json',[HERE/n for n in ['navigation_field.py','observation39.py','geometry.py','single_agent.py','expert_worker.py','NAVIGATION_UNIT_TESTS.json','FIELD_PATH_COMPARISON_TEST.json','PREREGISTRATION_EIGHT_NEIGHBOR_FIX.receipt.json']],before_expert_labels=True,old_designs_retained=True)
# The current training freeze incorporates the same corrected navigation source.
old=read(HERE/'TRAINING_PIPELINE39_V3.receipt.json');names=old['files']
write(HERE/'TRAINING_PIPELINE39_V4.receipt.json',dict(utc=utc(),before_demonstration_collection=True,before_training=True,files={p:sha(p) for p in names},reason='Strict eight-neighbor source correction before any data; all old receipts retained'))
write(HERE/'TRAINING_PIPELINE39_CURRENT.json',dict(receipt='TRAINING_PIPELINE39_V4.receipt.json',sha256=sha(HERE/'TRAINING_PIPELINE39_V4.receipt.json')))
for n in ['REPORT.md','HANDOFF.md','deliver_failed_expert_pilot.py']:
 p=HERE/n;p.write_text(p.read_text().replace('TRAINING_PIPELINE39_V3.receipt.json','TRAINING_PIPELINE39_V4.receipt.json'))
print('eight-neighbor exact profiles registered')
