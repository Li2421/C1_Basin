from common import *
from all_policy_diagnostics40 import analyze
for family in 'BF':analyze(family)
write(HERE/'PILOT_SAVED_DIAGNOSTICS40_COMPLETE.json',dict(utc=utc(),families=list('BF'),new_rollouts=0))
