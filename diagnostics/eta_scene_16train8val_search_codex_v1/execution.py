"""Dispatch immutable old cases to the isolated frozen numerical implementations."""
from common import *


def execute(c):
    if c.get('cross_scene_new'):
        from runtime_new import run
    elif c['state']['scene'].startswith('lit_swap8'):
        if c.get('scene_revision'):
            from swap_runtime_user_fix_20261023 import run
        else:
            from swap_runtime import run
    else:
        from runtime_correction_pa_memo_query_disk_candidate_spatial_mask_20261022 import run
    return run(c)
