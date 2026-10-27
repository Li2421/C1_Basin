"""Prevent duplicate in-flight UIDs between circle learned and selected-fixed eval."""
from experiment import *
from registry import verify_receipt
from atomic_write import write
import ast,difflib


def main():
    verify_receipt('ANALYSIS_SUPPLEMENT_READY_20261022.receipt.json')
    verify_receipt('ANALYSIS_SUPPLEMENT_DELIVERY_CLARIFIED_20261022.receipt.json')
    proof='CIRCLE_EVALUATION_REUSE_ORDER_20261022.receipt.json';assert not (HERE/proof).exists()
    assert not (HERE/'learning_eval_revised_20261021_lit_circle_five_PREFLIGHT.receipt.json').exists()
    assert not (HERE/'learning_eval_revised_20261021_trainfixed_lit_circle_five_PREFLIGHT.receipt.json').exists()
    copies={'scene_learning_scheduler_supplement_20261022.py':'scene_learning_scheduler_reuse_order_20261022.py','correction_continuation_supplement_clarified_20261022.py':'correction_continuation_reuse_order_20261022.py'};rows=[]
    for old,new in copies.items():
        before=(HERE/old).read_text();after=before
        for a,b in copies.items():after=after.replace(Path(a).stem,Path(b).stem)
        if new.startswith('scene_'):
            needle='            stage=stage_name(scene)\n'
            insert="            if not (HERE/f'TRAIN_SELECTED_FIXED_20261022_{scene}.receipt.json').exists():action('freeze_training_selected_fixed_'+scene,'analysis_standard_supplement_20261022.py')\n            if scene=='lit_circle_five' and not stage_done('learning_eval_revised_20261021_'+scene):continue\n"+needle
            assert after.count(needle)==1;after=after.replace(needle,insert)
            needle="                if not (HERE/f'{stage}_PREFLIGHT.receipt.json').exists():action('train_selected_fixed_plan_'+scene,'plan_training_selected_fixed_20261022.py',scene)"
            insert="                if scene=='lit_circle_five' and not (HERE/f'{stage}_PREFLIGHT.receipt.json').exists():action('merge_completed_circle_learned_before_fixed_preflight','db_merge.py')\n"+needle
            assert after.count(needle)==1;after=after.replace(needle,insert)
            after=after.replace("and all(stage_done(s) for s in fixed_evaluations)","and len(fixed_evaluations)==len(NAMES) and all(stage_done(s) for s in fixed_evaluations)")
        ast.parse(after);p=HERE/new;assert not p.exists();p.write_text(after)
        diff=p.with_suffix('.py.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new)))
        rows.append(dict(original=old,new=new,original_sha256=sha(HERE/old),new_sha256=sha(p),diff=str(diff)))
    write(HERE/'CIRCLE_EVALUATION_REUSE_ORDER_MAP_20261022.json',dict(utc=utc(),copies=copies,static_copies=rows,selection='Freeze after complete training bundle and before learned evaluation',execution_order='Complete learned circle evaluation;official merge;selected-fixed preflight;run only remaining genuinely missing identities',reason='Official preflight sees committed cache rows,not other stages in-flight requests;serialize overlapping exact UID scopes to avoid redundant physical execution',both_circle_evaluation_stages_not_yet_preflighted=True,science_or_statistics_changed=False))
    seal(proof,[Path(__file__),HERE/'CIRCLE_EVALUATION_REUSE_ORDER_MAP_20261022.json',HERE/'ANALYSIS_SUPPLEMENT_READY_20261022.receipt.json',HERE/'ANALYSIS_SUPPLEMENT_DELIVERY_CLARIFIED_20261022.receipt.json']+[p for r in rows for p in (HERE/r['new'],Path(r['diff']))],scientific_execution_semantics_unchanged=True,statistics_unchanged=True,prevent_duplicate_inflight_execution=True)
    print('PASS:circle evaluation reuse order sealed before either evaluation preflight',flush=True)


if __name__=='__main__':main()
