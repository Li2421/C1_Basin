from common import *
from all_policy_diagnostics40 import analyze
for family in 'ACDE':
    if (HERE/f'NAVIGATION_CAPABILITY_{family}.json').exists():analyze(family)
write(HERE/'ALL_SAVED_POLICY_DIAGNOSTICS40_COMPLETE.json',dict(utc=utc(),new_rollouts=0))
