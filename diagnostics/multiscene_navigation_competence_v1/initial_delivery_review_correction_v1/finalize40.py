"""Ordered read-only evidence review, ending audit and final report writing."""
from common import *
from submit_aux40 import wait_submit
import time
import xml.etree.ElementTree as ET


def main():
    while not (HERE/'CORRECTIVE_STUDY40_COMPLETE.json').exists():
        time.sleep(5)
    assert (HERE/'AF_ANALYSIS_FROZEN.receipt.json').exists()
    assert read(HERE/'LOCKED_G40_NOT_RUN.json')['corrective_evaluation_count']==0
    evidence=HERE/'EVIDENCE_REVIEW40_COMPLETE.receipt.json'
    if not evidence.exists():
        wait_submit('saved_evidence_review40','review_evidence40.py',workers=1,memory_gib=2,freeze_paths=[HERE/'CORRECTIVE_STUDY40_COMPLETE.json',HERE/'AF_ANALYSIS_FROZEN.receipt.json'])
    assert evidence.exists() and read(HERE/'EVIDENCE_REVIEW40.json')['passed']
    if not (HERE/'FIGURES40_PROVENANCE.json').exists():
        wait_submit('final_curves40','figures40.py',workers=1,memory_gib=2,freeze_paths=[HERE/'AF_ANALYSIS_FROZEN.receipt.json'])
    assert (HERE/'FIGURES40_PROVENANCE.json').exists()
    for path in list((HERE/'figures').glob('*.svg'))+list((HERE/'geometry_figures').glob('*.svg')):
        ET.parse(path)
    if not (HERE/'INTEGRITY_RESUME_END.json').exists():
        from audit_end40 import capture
        capture('RESUME_END')
    if not (HERE/'FINAL_DELIVERY40.receipt.json').exists():
        wait_submit('final_delivery40','delivery40.py',workers=1,memory_gib=2,freeze_paths=[HERE/'INTEGRITY_RESUME_END.json',evidence,HERE/'FIGURES40_PROVENANCE.json'])
    receipt=read(HERE/'FINAL_DELIVERY40.receipt.json')
    assert all(sha(p)==v for p,v in receipt['files'].items())
    result=read(HERE/'results.json')
    notes=read(HERE/'CURRENT_RUN_NOTES.json')
    notes.update(utc=utc(),current_phase='COMPLETE_WITH_GAPS',final_delivery_pending=[],evaluation_analysis_completed=True,final_receipt='FINAL_DELIVERY40.receipt.json',integrity_passed=result['integrity']['passed'],locked_G_evaluations=0)
    write(HERE/'CURRENT_RUN_NOTES.json',notes)
    seal('FINALIZATION40_COMPLETE.receipt.json',[HERE/'FINAL_DELIVERY40.receipt.json',HERE/'CURRENT_RUN_NOTES.json',HERE/'finalize40.py'],new_rollouts=0,G_evaluations=0,integrity_passed=result['integrity']['passed'])
    print('FINALIZATION40_COMPLETE',result['integrity']['passed'],flush=True)


if __name__=='__main__':
    main()
