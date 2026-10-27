from audit_supplement import *
import ast,subprocess,difflib
B=HERE/'backend';B.mkdir(exist_ok=True);native=HERE/'native_resume';native.mkdir(exist_ok=True)
records=[]
def copy(src,dst,code=None):
    h=evidence(src);dst.parent.mkdir(parents=True,exist_ok=True);old=src.read_bytes();new=old if code is None else code.encode();dst.write_bytes(new)
    records.append(dict(source=str(src),source_sha256=h,target=str(dst),target_sha256=sha(dst),changed=old!=new))
    if old!=new:(dst.with_suffix(dst.suffix+'.diff')).write_text(''.join(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile=str(src),tofile=str(dst))))
for src in (H/'native_resume').glob('*'):
    if src.suffix not in ('.hpp','.cpp'):continue
    code=src.read_text().replace(str(H/'runtime_cbs_query_cache_resume'),str(HERE/'runtime_cbs_query_cache'))
    copy(src,native/src.name,code)
command=['g++','-O3','-std=c++17','-shared','-fPIC',str(native/'cbs_kernel_query_disk_candidate_20261022.cpp'),'-o',str(native/'cbs_kernel_resume.so'),'-lcrypto']
subprocess.run(command,check=True)
for src in H.glob('*.so'):
    if 'cbs_kernel_query_disk' not in src.name:copy(src,HERE/src.name)
pending=['runtime_correction_pa_memo_query_disk_candidate_spatial_mask_20261022','failure_analysis','stats_kernel','analysis_kernels_resume'];seen=set()
while pending:
    name=pending.pop()
    if name in seen:continue
    seen.add(name)
    candidates=[H/'resume_backend'/(name+'.py'),V2/(name+'.py'),H/(name+'.py')]
    src=next((p for p in candidates if p.exists()),None)
    if src is None:continue
    if name in ('experiment','audit'):continue
    old=src.read_text();code=old
    for node in ast.walk(ast.parse(old)):
        names=[node.module] if isinstance(node,ast.ImportFrom) else [n.name for n in node.names] if isinstance(node,ast.Import) else []
        pending.extend(n.split('.')[0] for n in names if n)
    # Every imported module uses the root supplement experiment output boundary.
    if name=='legacy_states':
        code=code.replace("receipt=read(HERE/'CIRCLE_SPEED.receipt.json');v=read(HERE/'CIRCLE_SPEED.json')","receipt=read(HERE/'CIRCLE_SPEED.receipt.json');v=read(HERE/'CIRCLE_SPEED.json')")
        code=code.replace("str(HERE/'CIRCLE_SPEED.json')","str(V1/'CIRCLE_SPEED.json')") if False else code
    copy(src,B/src.name,code)
# The circle config checks the path key in its old receipt; make a new path-only receipt.
for name in ('CIRCLE_SPEED.json','CONTROLLER_MANIFEST.json','FIXED_ETA.json'):
    copy(V2/name,HERE/name)
write(HERE/'CIRCLE_SPEED.receipt.json',dict(files={str(HERE/'CIRCLE_SPEED.json'):sha(HERE/'CIRCLE_SPEED.json')}))
write(HERE/'BACKEND_BUILD.json',dict(utc=datetime.now(timezone.utc).isoformat(),sources=records,native_command=command,native_sha256=sha(native/'cbs_kernel_resume.so')))
print('Backend sources',len(records),flush=True)
