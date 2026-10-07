"""Transitive safety/Flow identity proof for archived FF compatibility."""
from .design import ROOT,MAIN,FIELD,read,sha,freeze


def main():
    snapshot=read(FIELD/'source_snapshot.json');files=[]
    for r in snapshot['files']:
        if r['path'].endswith('.py'):
            p=FIELD/'source'/r['path'];assert p.exists() and sha(p)==r['sha256'],r['path']
            files.append(dict(path=str(p),sha256=r['sha256']))
    equivalent=[]
    for rel in ('diagnostics/success_basin_multimodality/exact_projector.py',
        'diagnostics/double_bottleneck_eta3_basin/tools/exact_projection_retry.py','shared_control/hard_projection.py',
        'single_integrator/cbf.py','ring_exchange/safety.py','new_benchmark_common/macflow.py'):
        assert sha(MAIN/rel)==sha(FIELD/'source'/rel)
        equivalent.append(dict(main_path=str(MAIN/rel),copied_path=str(FIELD/'source'/rel),sha256=sha(MAIN/rel)))
    freeze(ROOT/'runtime_dependency_audit.json',dict(snapshot_sha256=sha(FIELD/'source_snapshot.json'),
        snapshot_created=snapshot['created_utc'],python_files_verified=len(files),files=files,
        main_vs_copied_import_equivalence=equivalent,modified_snapshot_files=0,
        meaning='Original copied source precedes archived FF outcomes and remains byte-identical; transitive projector/Flow helpers are also identical to main imports.'))
    print(dict(source_files_verified=len(files),transitive_import_equivalence=len(equivalent),changed=0))


if __name__=='__main__':main()
