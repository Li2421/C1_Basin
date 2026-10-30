"""Load the copied frozen learned actors; no generator/critic training or selection."""
from core import *

CODE = HERE / 'inputs/diagnostics/hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027/hetset/learned'


def install():
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import learned_actor
    assert Path(learned_actor.__file__).resolve().is_relative_to(CODE)
    learned_actor.install()


def rows():
    install()
    import learned_actor
    copied = read(HERE / 'INPUT_AUDIT.json')['models']
    result = []
    for model in copied:
        checkpoint = HERE / model['copied_checkpoint']
        for filename, expected in model['files'].items():
            assert sha(checkpoint / filename) == expected, ('Frozen learned weights changed', model['name'], filename)
        result.append(learned_actor.row(model['name'], str(checkpoint)))
    assert len(result) == 6
    return result
