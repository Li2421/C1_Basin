"""Round39 output boundary and authoritative database package."""
from core import *
EXP='exp_multiscene_navigation_competence_v1'
import shared_rollout_db
assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
from shared_rollout_db.src import rollout_db, cache_writer, planner
for module in [rollout_db,cache_writer,planner]:
    assert Path(module.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
assert (cache_writer.ROOT/'journals').resolve()==(ROOT/'shared_rollout_db/journals').resolve()

def frozen_config():
    prereg()
    if (HERE/'DESIGN_CONFIG_V2.receipt.json').exists():
        receipt=read(HERE/'DESIGN_CONFIG_V2.receipt.json')
        assert sha(HERE/'DESIGN_CONFIG_V2.json')==receipt['sha256']
        return read(HERE/'DESIGN_CONFIG_V2.json')
    receipt=read(HERE/'PREREGISTRATION.receipt.json')
    assert sha(HERE/'DESIGN_CONFIG.json')==receipt['files'][str((HERE/'DESIGN_CONFIG.json').resolve())]
    return read(HERE/'DESIGN_CONFIG.json')
