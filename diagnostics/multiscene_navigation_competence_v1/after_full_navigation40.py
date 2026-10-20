"""Continue stage order once full navigation is complete; one merger coordinator."""
from common import *
import time

def main():
    while not (HERE/'FULL_NAVIGATION_PHASE_COMPLETE.json').exists():time.sleep(5)
    phase=read(HERE/'FULL_NAVIGATION_PHASE_COMPLETE.json')
    from full_descriptive_pipeline40 import main as descriptions
    descriptions()
    if not phase['correction_stage_eligible']:
        write(HERE/'CORRECTIVE_STAGE40_BLOCKED.json',dict(utc=utc(),reason='Fewer than four first-six families passed navigation capability',navigation_families=phase['eligible_first_six_families'],G_corrective_locked=True));return
    receipt=read(HERE/'MULTIAGENT_CAPABILITY40_IMPLEMENTATION_FROZEN_V2.receipt.json')
    assert all(sha(path)==expected for path,expected in receipt['files'].items())
    from multiagent_capability_pipeline40 import main as capability
    capability()
    write(HERE/'AFTER_FULL_NAVIGATION40_COMPLETE.json',dict(utc=utc(),next_stage='Review capability gaps then run previously frozen probe and source-only selection design',G_corrective_locked=True))

if __name__=='__main__':main()
