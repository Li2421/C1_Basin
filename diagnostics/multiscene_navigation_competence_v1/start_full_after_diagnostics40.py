"""One submission/merger coordinator at a time; full stage begins after pilot diagnostics."""
from common import *
import time
while not (HERE/'PILOT_DESCRIPTIVE_EXECUTION_COMPLETE.json').exists():time.sleep(5)
assert read(HERE/'PILOT_DECISION_FROZEN.json')['passed']
from full_navigation_pipeline40 import main
main()
