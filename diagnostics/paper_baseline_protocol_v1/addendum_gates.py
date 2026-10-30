"""Prospective gates for the2026-10-30addendum; no simulated outcomes."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def native_methods():
    path=HERE/'NATIVE_REPRODUCTION_RESULTS.json'
    assert path.exists(), 'Native author-scene reproduction required before port edits'
    value=json.loads(path.read_text())
    assert value.get('complete'), 'Native author-scene reproduction incomplete'
    methods=[m for m,r in value['methods'].items() if r['reproducible']]
    assert methods, 'All three native reproductions failed: stop and report to user'
    return methods

def require_before_menus():
    native_methods()
    for name in ['PORT_FIX_BEFORE_AFTER.json','NARROW_PASSAGE_DIAGNOSTIC.json','CONSISTENT_BOX_SCENES.json','GOAL_ALIGNMENT_CHECK.json']:
        path=HERE/name
        assert path.exists() and json.loads(path.read_text()).get('complete'), 'Required addendum phase incomplete: '+name
    assert (HERE/'MARGIN_ALIGNMENT.md').exists()
