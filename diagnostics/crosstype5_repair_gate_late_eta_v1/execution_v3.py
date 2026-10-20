"""Explicit arm dispatch; historical given loop remains byte-identical."""
def execute(c,p,cache,phase,probe=False):
    if c.get('late_source','given')=='requery':
        from requery_execution_v3 import execute as run
    else:
        from new_execution import execute as run
    return run(c,p,cache,phase,probe=probe)
