import dataclasses
from diagnostics.controller_state_transfer_v1.runtime import base_runtime as original_base,controller,rollout

def base_runtime(width=None):
 base=original_base(width);base.config=dataclasses.replace(base.config,max_steps=400);return base
