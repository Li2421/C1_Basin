"""Finalize reproducible audit manifest and check immutable source artifacts."""
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=ROOT/'diagnostics/true_q_geometry'; previous=json.loads((old/'manifest.json').read_text())
    for group in ['frozen_sources_and_specs','study_code','raw_indexes','deliverables_and_figures']:
        for r in previous[group]:assert digest(ROOT/r['path'])==r['sha256'],r['path']
    fresh=json.loads((HERE/'fresh_replication.json').read_text())
    assert fresh['new_rollouts']==192 and fresh['all_three_replicate']
    assert digest(HERE/'replication_plan.json')==fresh['plan_sha256']
    for r in fresh['records']:assert digest(HERE/r['relative_path'])==r['sha256']
    needed=['astra_independent_audit.md','control_geometry_audit.json','nonsmoothness_audit.json',
        'multimodality_audit.json','timeout_substitution_audit.json','projection_audit.json','fresh_replication.json']
    for name in needed:assert (HERE/name).is_file()
    files=[]
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json':
            if p.suffix=='.json':json.loads(p.read_text())
            files.append({'path':str(p.relative_to(ROOT)),'sha256':digest(p),'bytes':p.stat().st_size})
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    manifest={'schema':'astra_independent_true_q_audit_v1','completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'decision':json.loads((HERE/'decision.json').read_text()),'git_head':head.stdout.strip() if head.returncode==0 else None,
        'dirty_tree_note':'Use source and data hashes; no commit created.',
        'old_manifest_sha256':digest(old/'manifest.json'),'old_tree_unchanged_against_manifest':True,
        'frozen_inputs':previous['frozen_sources_and_specs']+[previous['frozen_checkpoint'],previous['protocol']],
        'new_full_continuations':192,'actual_new_physical_steps':fresh['actual_physical_steps'],
        'device':fresh['device'],'independent_orchestration':'replicate.py uses frozen primitives; does not call old rollout runner',
        'blindness_limit':'Previous conclusion available in conversation; narrative read only after blind_static_conclusion.json locked.',
        'optional_basin_grid':'Skipped to keep audit focused; no topology claim made.',
        'validation':json.loads((HERE/'verification.json').read_text()),'files':files}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print({'manifest':'PASS','indexed_files':len(files),'new_rollouts':192,'new_physical_steps':fresh['actual_physical_steps']})

if __name__=='__main__':main()
