"""Freeze the full local execution dependency set before any missing rollout."""
from common import *

DEPENDENCIES=('worker.py','rollout_runtime.py','zoo_runtime.py','zoo_models.py','ippo.py',
              'prefix_reuse.py','probe_prefix.py','history_audit.py','controllers.py','historical_controllers.py',
              'legacy_controllers.py','learned.py','gaussian.py','right_hand.py','trigger.py','progress.py',
              'common.py','core.py','PROBE_PREFIX_INVENTORY.json','HISTORY_REUSE_CANDIDATES.json')


def freeze(stage,**metadata):
    seal(f'{stage}_EXECUTION_FROZEN.receipt.json',
         [HERE/f'{stage}_protocol.json',HERE/f'{stage}_CASES.json',HERE/f'{stage}_PREFLIGHT.receipt.json',
          *[HERE/name for name in DEPENDENCIES]],before_rollouts=True,**metadata)
