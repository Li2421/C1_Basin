from common import *
from submit import phase
assert (HERE/'CHOICES_FROZEN.receipt.json').exists();assert (HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json').exists()
phase('main',nextstep='main_done')
