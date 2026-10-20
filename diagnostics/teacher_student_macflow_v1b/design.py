"""Register new round30b identities only; retain original round30 state pairing."""
from core import *
from registration import *
from audit_utils import source_inventory
import jax
jax.config.update('jax_enable_x64',False)
def initial():
    assert (HERE/'TRAINING_FROZEN.receipt.json').exists()
    assert read(HERE/'ROUND28_REPLAY_GATE.json')['completed']==16 and read(HERE/'ROUND28_REPLAY_GATE.json')['passed']
    register('initial',[exp_row(p) for p in ['pilot','probe','main']])
    # All states already have exact registered identities from round30; validate without reinserting.
    with connect(True) as db:
        for s in read(HERE/'PILOT_STATES.json')+read(HERE/'EVAL_STATES.json'):
            r=db.execute('SELECT * FROM state WHERE state_uid=?',(s['state_uid'],)).fetchone();assert r and r['physical_state_json']==canonical(s['physical']) and r['content_hash']==s['content_hash']
    source_inventory('imports_design.json')
if __name__=='__main__':initial()
