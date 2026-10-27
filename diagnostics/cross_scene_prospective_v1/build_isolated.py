"""Copy the completed historical execution graph; relocate only native cache roots."""
from pathlib import Path
import hashlib, json, os, subprocess, difflib

H = Path(__file__).resolve().parent
R = H.parents[1]
OLD = R / 'diagnostics/cross_scene_correction_v1'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert (OLD/'COMPLETE.json').exists()
    assert json.loads((OLD/'INTEGRITY_COMPARE.json').read_text())['passed']
    rows = []
    def copy(src, dst, text=None):
        before = src.read_bytes(); data = before if text is None else text.encode()
        assert dst.resolve().is_relative_to(H)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists(): assert dst.read_bytes() == data, dst
        else: dst.write_bytes(data)
        rows.append(dict(source=str(src), source_sha256=sha(src), target=str(dst),
                         target_sha256=sha(dst), changed=before != data))
        if before != data:
            dst.with_suffix(dst.suffix+'.diff').write_text(''.join(difflib.unified_diff(
                before.decode().splitlines(True),data.decode().splitlines(True),fromfile=str(src),tofile=str(dst))))
    for name in ('experiment.py','bootstrap.py','execution.py','models.py','scenes.py',
                 'scene_environment.py','controllers_new.py','runtime_new.py','runtime_checkpointed.py',
                 'checkpoint_store.py','CIRCLE_SPEED.json','CONTROLLER_MANIFEST.json','FIXED_ETA.json',
                 'LAYOUT_REVISION_DESIGN.json','MOVING_USER_FIX_UNIT_GATE.json'):
        copy(OLD/name,H/name)
    # This is a new coordinator; the copied numerical modules above are byte-identical.
    code=(OLD/'common.py').read_text().replace("EXP = 'exp_cross_scene_correction_v1'", "EXP = 'exp_cross_scene_prospective_v1'")
    code=code.replace("'Cross scene correction v1'", "'Cross scene prospective v1'")
    code=code.replace("'PROTOCOL_AMENDMENT.md'", "'PROTOCOL.md'")
    code=code.replace("'EVIDENCE_SHA256_START.json'", "'REUSED_EVIDENCE_SHA256_START.json'")
    copy(OLD/'common.py',H/'common.py',code)
    for src in (OLD/'backend').glob('*.py'): copy(src,H/'backend'/src.name)
    for src in OLD.glob('*.so'): copy(src,H/src.name)
    for src in (OLD/'native_resume').iterdir():
        if src.suffix in ('.cpp','.hpp'):
            copy(src,H/'native_resume'/src.name,src.read_text().replace(
                str(OLD/'runtime_cbs_query_cache'),str(H/'runtime_cbs_query_cache')))
    copy(R/'docs/prompt_cross_scene_prospective_v1_20261025.md',H/'PROTOCOL.md')
    (H/'CIRCLE_SPEED.receipt.json').write_text(json.dumps(dict(files={str(H/'CIRCLE_SPEED.json'):sha(H/'CIRCLE_SPEED.json')}),indent=2)+'\n')
    (H/'tmp').mkdir(exist_ok=True)
    env=dict(os.environ,TMPDIR=str(H/'tmp'))
    command=['g++','-O3','-std=c++17','-shared','-fPIC',str(H/'native_resume/cbs_kernel_query_disk_candidate_20261022.cpp'),
             '-o',str(H/'native_resume/cbs_kernel_resume.so'),'-lcrypto']
    subprocess.run(command,env=env,check=True)
    changed=[r for r in rows if r['changed'] and '/native_resume/' in r['target']]
    assert len(changed)==1 and changed[0]['target'].endswith('cbs_query_disk_memo_candidate_20261022.hpp')
    (H/'BACKEND_BUILD.json').write_text(json.dumps(dict(sources=rows,native_command=command,
        native_sha256=sha(H/'native_resume/cbs_kernel_resume.so'),native_only_change='cache root'),indent=2)+'\n')
    inputs={r['source']:r['source_sha256'] for r in rows}
    for d in (R/'shared_control',R/'single_integrator',R/'bottleneck_family',R/'shared_rollout_db/src',R/'diagnostics/double_bottleneck_eta3_basin/tools'):
        for p in d.glob('*.py'): inputs[str(p)]=sha(p)
    for name in ('LABEL_TABLES.json','SCENES.json','FEATURES.json','STATISTICS.json','LOSO_CHOICES.json','LOFO_CHOICES.json','REPORT.md','COMPLETE.json','INTEGRITY_COMPARE.json'):
        inputs[str(OLD/name)]=sha(OLD/name)
    for p in (R/'diagnostics/cross_scene_homogeneity_audit_v1').glob('*'):
        if p.is_file():inputs[str(p)]=sha(p)
    for p in [R/'diagnostics/baseline_literature_scenes_distributed_v2/LEARNING_SPLIT_REVISED_20261021_lit_crowded_random.json',
              R/'diagnostics/baseline_literature_scenes_distributed_v2/LEARNING_SPLIT_REVISED_20261021_lit_narrow_passage.json',
              R/'diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/ACTIONS.json',R/'AGENTS.md',R/'shared_rollout_db/README.md']:
        inputs[str(p)]=sha(p)
    (H/'REUSED_EVIDENCE_SHA256_START.json').write_text(json.dumps(dict(files=inputs),indent=2)+'\n')
    print('Built isolated graph:',len(rows),'copies;',len(inputs),'audited inputs',flush=True)

if __name__=='__main__':main()
