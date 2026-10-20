import pathlib,hashlib,json,difflib
H=pathlib.Path(__file__).resolve().parent;R=H.parents[1];rows=json.loads((H/'COPY_PROVENANCE.json').read_text());patch=[]
for src,dst in [(R/'diagnostics/teacher_student_macflow_v1d/registration.py','registration.py'),(R/'diagnostics/ring_controller_critic_v1/fit.py','critic_fit.py')]:
 a=src.read_text();b=a
 if dst=='critic_fit.py':
  for line in ['os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")','os.environ.setdefault("OMP_NUM_THREADS", "1")','os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")']:b=b.replace(line,'# Environment configured by round30f core.')
 (H/dst).write_text(b);rows.append(dict(source=str(src),source_sha256=hashlib.sha256(a.encode()).hexdigest(),destination=str(H/dst),copy_sha256=hashlib.sha256(b.encode()).hexdigest()));patch.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile=str(src),tofile=str(H/dst)))
(H/'COPY_PROVENANCE.json').write_text(json.dumps(rows,indent=2));p=H/'COPY_DIFF.patch';p.write_text(p.read_text()+''.join(patch))
