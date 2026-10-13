"""Assert core provenance, denominators and frozen-rule reproducibility."""
from __future__ import annotations
import csv,json,xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent

def csvrows(name):
    with (HERE/name).open(newline='') as f:return list(csv.DictReader(f))

def main():
    toy=csvrows('toy_loss_analysis.csv');ring=csvrows('ring_gain_analysis.csv')
    pred=csvrows('heldout_prediction.csv');events=csvrows('event_pair_audit.csv')
    critical=csvrows('critical_event_rows.csv');transfer=csvrows('initial_action_decomposition.csv')
    assert len(toy)==len(ring)==384
    assert Counter(r['b15_outcome'] for r in toy)==Counter(loss=178,gain=10,both_success=25,both_failure=171)
    assert Counter(r['b15_outcome'] for r in ring)==Counter(gain=48,loss=1,both_success=29,both_failure=306)
    assert len(pred)==768 and len(events)==12288 and len(critical)==2255 and len(transfer)==768
    assert sum(r['scene']=='toy_give_way' and r['new_termination']=='deadlock' for r in critical)==1479
    assert sum(r['scene']=='ring_exchange' and r['TT_individual_goal_ever_count']!='4' for r in critical)==398
    assert json.loads((HERE/'extraction_summary.json').read_text())['toy_monitor_replay_all_match']
    assert json.loads((HERE/'extraction_summary.json').read_text())['seed0_valid_pairs']==767
    assert json.loads((HERE/'verification.json').read_text())['evidence_integrity']['new_full_continuations']==0
    for p in (HERE/'figures').glob('*.svg'):ET.parse(p)
    assert len(list((HERE/'figures').glob('*.svg')))==4
    assert len(csvrows('field_substep_probe.csv'))==44
    assert len(csvrows('horizon_ablation.csv'))==27
    substeps=json.loads((HERE/'field_substep_summary.json').read_text())
    assert max(float(r['new_action_reconstruction_error']) for r in substeps)<1e-8
    assert max(float(r['cached_old_action_error']) for r in substeps)<1e-8
    assert max(float(r['cached_new_action_error']) for r in substeps)<1e-8
    print('PASS: paired denominators, B15 groups, event counts, monitor replay, frozen predictions, SVGs')

if __name__=='__main__':main()
