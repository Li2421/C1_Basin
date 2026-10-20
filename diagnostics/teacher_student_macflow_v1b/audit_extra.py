from core import *
from audit_utils import source_inventory
from shared_rollout_db.src.rollout_db import connect
import ast,difflib

def main():
    modules=[]
    for f in sorted(HERE.glob('imports_*.json')):
        v=read(f)['imported_sources'];pkg=[p for p in v if p.endswith('/shared_rollout_db/__init__.py')];writer=[p for p in v if p.endswith('/shared_rollout_db/src/cache_writer.py')]
        assert pkg==[str(ROOT/'shared_rollout_db/__init__.py')] and writer==[str(ROOT/'shared_rollout_db/src/cache_writer.py')],f
        modules.append(dict(inventory=f.name,package=pkg[0],writer=writer[0],package_sha256=v[pkg[0]],writer_sha256=v[writer[0]]))
    with connect(True) as db:
        sources=[dict(r) for r in db.execute('SELECT * FROM source_file WHERE experiment_uid LIKE ?',('exp_teacher_student_macflow_v1b_%',))]
    assert all(Path(r['path']).resolve().is_relative_to(ROOT/'shared_rollout_db/journals') for r in sources)
    write(HERE/'DB_PATH_AUDIT.json',dict(utc=utc(),passed=True,expected_journals=str(ROOT/'shared_rollout_db/journals'),entry_guard='core.py imports official DB first and asserts package, writer and default journal path before directory/file creation; run.py reasserts and validates returned journal path',entry_inventories=modules,official_sources=sources))
    provenance=read(HERE/'COPY_PROVENANCE.json');patch=[]
    for row in provenance:
        src=Path(row['source']);dst=HERE/row['destination'];assert sha(src)==row['source_sha256'];row['destination_sha256']=sha(dst);patch.extend(difflib.unified_diff(src.read_text().splitlines(True),dst.read_text().splitlines(True),fromfile=str(src),tofile=str(dst)))
    write(HERE/'COPY_PROVENANCE.json',provenance);(HERE/'COPY_DIFF.patch').write_text(''.join(patch))
    refs=[D/'decentralized_pipeline_v1'/n for n in ['pipeline.py','run.py','core.py']];write(HERE/'ROUND28_COPY_AUDIT.json',dict(original_sources={str(p):sha(p) for p in refs},round30_intermediate_sources=read(D/'teacher_student_macflow_v1/COPY_PROVENANCE.json'),checks=read(HERE/'RESULTS_VALIDATION.json')['checks'],allowed_differences='paths/environment and experiment output namespace; decentralized_action and episode AST unchanged; new extra arms copied from round30; student chart and target/noise transformed locally'))
    source_inventory('imports_audit_extra.json')
if __name__=='__main__':main()
