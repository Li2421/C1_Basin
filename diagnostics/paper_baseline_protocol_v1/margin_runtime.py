"""Map explicit physical safety-distance constants to the common plant."""
import sys


def align_mgr(env):
    # The common configurations have zero additional pair margin. Static-wall
    # radius/margin are already represented by obstacle geometry in the port.
    assert env.config.agent_collision_margin == 0., 'Nonzero pair margin needs a distinct documented mapping'
    changed = []
    for name in ('params', 'agent', 'cbf', 'simulator', 'collision'):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, 'SAFE_RATIO'):
            module.SAFE_RATIO = 1.
            changed.append(name)
    assert 'cbf' in changed
    return changed
