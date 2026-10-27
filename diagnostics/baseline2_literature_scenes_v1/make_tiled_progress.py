"""Only add stdout step heartbeats to the already byte-replayed executor."""
from experiment import *
import ast


def main():
    source=HERE/'flow_runtime_tiled.py';text=source.read_text()
    anchor='    for t in range(cfg.max_steps):\n'
    added="        if t%100==0:print('BASELINE2_STEP',c['state']['index'],c['controller_payload']['nominal']['controller'],c['seed'],t,flush=True)\n"
    assert text.count(anchor)==1
    actual=HERE/'flow_runtime_tiled_progress.py';actual.write_text(text.replace(anchor,anchor+added))
    class RemoveHeartbeat(ast.NodeTransformer):
        removed=0
        def visit_If(self,node):
            if any(isinstance(x,ast.Constant) and x.value=='BASELINE2_STEP' for x in ast.walk(node)):
                assert len(node.body)==1 and isinstance(node.body[0],ast.Expr)
                call=node.body[0].value
                assert isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='print'
                assert not node.orelse
                self.removed+=1
                return None
            return self.generic_visit(node)
    visitor=RemoveHeartbeat();stripped=visitor.visit(ast.parse(actual.read_text()))
    assert visitor.removed==1
    assert ast.dump(stripped,include_attributes=False)==ast.dump(ast.parse(text),include_attributes=False)
    write(HERE/'TILED_PROGRESS_IO_PROOF.json',dict(
        utc=utc(),source=str(source),source_sha256=sha(source),actual=str(actual),actual_sha256=sha(actual),
        exact_AST_equal_after_removing_one_readonly_stdout_heartbeat=True,
        numeric_execution_unchanged=True,source_full_replay_receipt='TILED_PARENT_EQUIVALENCE.receipt.json'))
    seal('TILED_PROGRESS_IO.receipt.json',[source,actual,HERE/'TILED_PROGRESS_IO_PROOF.json',
         HERE/'make_tiled_progress.py',HERE/'TILED_PARENT_EQUIVALENCE.receipt.json'],
         simulation_and_controller_computations_unchanged=True)
    old=(HERE/'worker_interior_stream.py').read_text()
    new=old.replace('from flow_runtime_interior import run','from flow_runtime_tiled_progress import run')
    needle="    interior=read(HERE/'CONTROLLER_INTERIOR_EQUIVALENCE.receipt.json');assert all(sha(p)==h for p,h in interior['files'].items())"
    new=new.replace(needle,needle+"\n    for name in ('TILED_PARENT_EQUIVALENCE.receipt.json','TILED_PROGRESS_IO.receipt.json'):\n        proof=read(HERE/name);assert all(sha(p)==h for p,h in proof['files'].items())")
    new=new.replace("actual='flow_runtime_interior.py',proof='CONTROLLER_INTERIOR_EQUIVALENCE.receipt.json',source_sha256=sha(HERE/'flow_runtime_interior.py')",
                    "actual='flow_runtime_tiled_progress.py',proof='TILED_PROGRESS_IO.receipt.json',source_sha256=sha(HERE/'flow_runtime_tiled_progress.py')")
    new=new.replace("str(HERE/'worker_interior_stream.py')","str(HERE/'worker_tiled_stream.py')")
    (HERE/'worker_tiled_stream.py').write_text(new)
    (HERE/'flow_tiled.sbatch').write_text((HERE/'flow_interior.sbatch').read_text().replace('worker_interior_stream.py','worker_tiled_stream.py'))


if __name__=='__main__':main()
