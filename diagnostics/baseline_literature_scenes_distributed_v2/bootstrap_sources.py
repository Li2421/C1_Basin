"""Copy frozen inputs once; keep historical execution logic unchanged."""
from experiment import *
from source_inputs import copy,verify
import difflib

def main():
    mapping={name:name for name in (
        'controllers.py','controllers_interior.py','certified_planning.py',
        'planning.py','projection_adapter.py','projection_memo.py',
        'interior_nominal.py','official_sampler.py',
        'astar_kernel.so','cbs_kernel.so','astar_kernel_certified.so','cbs_kernel_certified.so',
        'astar_kernel.cpp','cbs_kernel.cpp','astar_kernel_certified.cpp','cbs_kernel_certified.cpp',
        'CONTROLLER_MANIFEST.json','FIXED_ETA.json','LEARNING_MENU.json','CIRCLE_SPEED.json',
        'capability_v2_lit_narrow_passage_QUALIFICATION.json',
        'capability_lit_doorway_QUALIFICATION.json','capability_lit_circle_five_QUALIFICATION.json')}
    mapping.update({'states.py':'legacy_states.py','flow_scene.py':'legacy_flow_scene.py',
                    'flow_runtime.py':'central_reference.py',
                    'flow_runtime_interior.py':'central_runtime.py',
                    'failure_analysis.py':'legacy_failure_analysis.py'})
    for source,target in mapping.items():copy(V1/source,HERE/target)
    copy(ROUND28/'pipeline.py',HERE/'source_snapshots/round28_pipeline.py')
    source=ROOT.parent/'Basin_C1_flow_field_poc_20261004/poc/field_controller.py'
    target=copy(source,HERE/'poc_field_controller.py')
    old=target.read_text()
    new=old.replace('ROOT = Path(__file__).resolve().parents[1]',
                    'ROOT = Path("/home/zhihan/research/Basin_C1_flow_field_poc_20261004")')
    new=new.replace('SOURCE = ROOT / "source"','SOURCE = Path("/home/zhihan/research/Basin_C1")')
    assert new!=old
    target.write_text(new)
    (HERE/'COPY_PATH_ONLY.diff').write_text(''.join(difflib.unified_diff(
        old.splitlines(True),new.splitlines(True),fromfile=str(source),tofile=str(target))))
    seal('POC_PATH_COPY.receipt.json',[target,HERE/'COPY_PATH_ONLY.diff'],
         source=str(source),source_sha256=sha(verify(source)),changes='Only ROOT/SOURCE path constants; native projection/retry/certificate logic unchanged.')
    # A new receipt for the byte-identical copied speed value. Do not alter or
    # counterfeit the original v1 receipt or its absolute path keys.
    seal('CIRCLE_SPEED.receipt.json',[HERE/'CIRCLE_SPEED.json'],
         source=str(V1/'CIRCLE_SPEED.json'),source_sha256=sha(verify(V1/'CIRCLE_SPEED.json')),
         original_receipt=str(V1/'CIRCLE_SPEED.receipt.json'),
         original_receipt_sha256=sha(verify(V1/'CIRCLE_SPEED.receipt.json')),
         computation='Exact frozen value reused; no LQR recomputation or rollout.')
    seal('SOURCE_COPIES.receipt.json',[HERE/'COPIED_SOURCES.jsonl',HERE/'COPY_PATH_ONLY.diff',HERE/'bootstrap_sources.py'])

if __name__=='__main__':main()
