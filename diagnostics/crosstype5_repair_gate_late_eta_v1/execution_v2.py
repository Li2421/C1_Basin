"""Explicit arm dispatch; historical given loop remains byte-identical."""
from requery_generator_v2 import FrozenMean  # Load reference modules before nominal-source path setup.

def execute(c,p,cache,phase,probe=False):
    if c.get('late_source','given')=='requery':
        from requery_execution_v2 import execute as run
    else:
        from new_execution import execute as run
    return run(c,p,cache,phase,probe=probe)
