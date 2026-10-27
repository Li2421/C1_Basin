"""Derive a static resumable loop; retain a machine-checkable exact source inverse."""
from common import *
import difflib
import ast


NAMES=('env','actor','positions','velocities','nominal','safe','actions','rewrite','clear',
       'geometry','certificates','durations','clear_types','switch','error','termination','d0')


def main():
    source=(HERE/'runtime_new.py').read_text()
    edits=[('def run(c):','def run(c,checkpoint=None):')]
    anchor="    for t in range(cfg.max_steps):"
    initialize="    elapsed_before=0.;next_step=0\n    restored=checkpoint.load()if checkpoint is not None else None\n    if restored is not None:\n"
    initialize+=''.join(f"        {n}=restored[{n!r}]\n"for n in NAMES)
    initialize+="        elapsed_before=restored['elapsed_before'];next_step=restored['next_step'];cfg=env.config\n    for t in range(next_step,cfg.max_steps):"
    edits.append((anchor,initialize))
    anchor="    violation=sum(any(d['maximum_violation']"
    snapshot=','.join(f'{n}={n}'for n in NAMES)
    inserted="        if checkpoint is not None:\n            checkpoint.step(t+1,lambda:dict("+snapshot+",elapsed_before=elapsed_before+time.monotonic()-start))\n"
    edits.append((anchor,inserted+anchor))
    edits.append(('elapsed_seconds=time.monotonic()-start,','elapsed_seconds=elapsed_before+time.monotonic()-start,'))
    edits.append(('    return result,arrays','    if checkpoint is not None:result[\'operational_resume\']=checkpoint.provenance()\n    return result,arrays'))
    output=source
    for old,new in edits:
        assert output.count(old)==1,old
        output=output.replace(old,new,1)
    recovered=output
    for old,new in reversed(edits):
        assert recovered.count(new)==1,new
        recovered=recovered.replace(new,old,1)
    assert recovered==source
    ast.parse(output)
    (HERE/'runtime_checkpointed.py').write_text(output)
    (HERE/'CHECKPOINT_RUNTIME.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),output.splitlines(True),fromfile='canonical runtime_new.py',tofile='derived runtime_checkpointed.py')))
    write(HERE/'CHECKPOINT_BUILD.json',dict(utc=utc(),baseline=str(HERE/'runtime_new.py'),baseline_sha256=sha(HERE/'runtime_new.py'),
        derived=str(HERE/'runtime_checkpointed.py'),derived_sha256=sha(HERE/'runtime_checkpointed.py'),
        inverse_text_transform_exact=True,edits=edits,
        scope='Operational pause/resume only; original numerical statements, actor, physics, RNG and success criteria retained. No old module monkeypatch.'))


if __name__=='__main__':main()
