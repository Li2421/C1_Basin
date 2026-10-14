from core import *
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
prereg();native=read(ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json');expected=generate('round14a_ring_independent',64,2026142400,native);actual=read(HERE/'FRESH_STATES.json');checks=[]
for a,b in zip(actual,expected):
 assert a['physical']==b['physical'] and a['state_uid']==b['state_uid'] and a['content_hash']==b['content_hash']
 checks.append(dict(state_uid=a['state_uid'],physical_content_identical=True))
write(HERE/'STATE_GENERATOR_EQUIVALENCE.json',dict(utc=utc(),passed=True,checks=checks,official_generator_sha256=sha(P1/'state_generator.py'),sampling_config_and_physical_draws_identical=True,alias_rule='task-specific alias and its SHA-derived RNG namespace frozen in FRESH_STATES and protocol before main; aliases differ from generic generator, no state redrawing or target-label use',success_labels_read=False))
print('Round10 native generator physical draws and state identities exact64 PASS')
