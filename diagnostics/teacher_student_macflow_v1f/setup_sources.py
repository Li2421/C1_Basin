import pathlib,hashlib,json,difflib
H=pathlib.Path(__file__).resolve().parent;R=H.parents[1];F=pathlib.Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
rows=[];diff=[]
def copy(src,dst,replacements=()):
 src=pathlib.Path(src);dst=H/dst;dst.parent.mkdir(parents=True,exist_ok=True);a=src.read_text();b=a
 for old,new in replacements:
  assert old in b,(src,old);b=b.replace(old,new)
 dst.write_text(b);rows.append({'source':str(src),'source_sha256':hashlib.sha256(a.encode()).hexdigest(),'destination':str(dst),'copy_sha256':hashlib.sha256(b.encode()).hexdigest()});diff.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile=str(src),tofile=str(dst)))
for name in ['core.py','pipeline.py','episode_frozen.py','fair_rhr.py','audit_utils.py']:
 copy(R/'diagnostics/teacher_student_macflow_v1d'/name,name)
# A shadow package changes only source paths and environment defaults. No runtime monkeypatch.
pkg=H/'vendor/new_benchmark_common';pkg.mkdir(parents=True,exist_ok=True)
(pkg/'__init__.py').write_text('__path__.append("'+str(F/'source/new_benchmark_common')+'")\n')
copy(F/'source/new_benchmark_common/macflow.py','vendor/new_benchmark_common/macflow.py',[("REPO_ROOT = Path(__file__).resolve().parents[1]","REPO_ROOT = Path('"+str(F/'source')+"')"),("os.environ.setdefault(\"JAX_PLATFORMS\", \"cpu\")","# Environment configured by this round's entry point.")])
# Core only path/env modifications; pre-load correct DB and copied side-effecting module.
p=H/'core.py';a=p.read_text();b=a.replace("sys.path[:0]=[str(ROOT),str(FIELD),str(FIELD/'source')]","sys.path[:0]=[str(HERE/'vendor'),str(FIELD/'source'),str(FIELD),str(ROOT)]\nimport new_benchmark_common.macflow")
p.write_text(b);rows[0]['copy_sha256']=hashlib.sha256(b.encode()).hexdigest();diff.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='copied core',tofile=str(p)))
for name,src in [('TEACHERS.json','teacher_student_macflow_v1b'),('STUDENTS.json','teacher_student_macflow_v1d'),('EVAL_STATES.json','teacher_student_macflow_v1d'),('PILOT_STATES.json','teacher_student_macflow_v1'),('TRAINING_CONFIG.json','teacher_student_macflow_v1')]:copy(R/'diagnostics'/src/name,name)
(H/'COPY_PROVENANCE.json').write_text(json.dumps(rows,indent=2));(H/'COPY_DIFF.patch').write_text(''.join(diff))
