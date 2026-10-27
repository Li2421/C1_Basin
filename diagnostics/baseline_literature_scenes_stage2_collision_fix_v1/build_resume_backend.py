"""Static private import graph, native cache path rebuilt; no module monkeypatch."""
from audit import HERE,sha,write
import ast,json,subprocess,difflib

def main():
    backend=HERE/'resume_backend';native=HERE/'native_resume'
    backend.mkdir(exist_ok=True);native.mkdir(exist_ok=True)
    assert not (backend/'BUILD_RECEIPT.json').exists()
    records=[]
    for src in HERE.glob('*.hpp'):
        dst=native/src.name;code=src.read_text()
        if src.name=='cbs_query_disk_memo_candidate_20261022.hpp':
            code=code.replace('/home/zhihan/research/Basin_C1/diagnostics/baseline_literature_scenes_distributed_v2/runtime_cbs_query_memo_candidate_20261022',str(HERE/'runtime_cbs_query_cache_resume'))
        dst.write_text(code);records.append(dict(source=str(src),sha256=sha(src),target=str(dst),target_sha256=sha(dst)))
    cpp=HERE/'cbs_kernel_query_disk_candidate_20261022.cpp';dst=native/cpp.name;dst.write_bytes(cpp.read_bytes())
    library=native/'cbs_kernel_resume.so'
    command=['g++','-O3','-std=c++17','-shared','-fPIC',str(dst),'-o',str(library),'-lcrypto']
    subprocess.run(command,check=True)
    pending=['runtime_fix','dynamics_spatial_mask_20261022','stage2_diagnostics','unit_gates','snapshot_gate','registry_fix'];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);src=HERE/(name+'.py')
        if not src.exists():continue
        old=src.read_text();code=old
        for node in ast.walk(ast.parse(old)):
            names=[node.module] if isinstance(node,ast.ImportFrom) else [n.name for n in node.names] if isinstance(node,ast.Import) else []
            pending.extend(n.split('.')[0] for n in names if n and (HERE/(n.split('.')[0]+'.py')).exists())
        if name=='experiment':code=code.replace('HERE = Path(__file__).resolve().parent','HERE = Path(__file__).resolve().parent.parent')
        if name=='tiled_planning_pa_memo_query_disk_candidate_spatial_mask_20261022':
            code=code.replace("HERE/'cbs_kernel_query_disk_candidate_20261022.so'","HERE/'native_resume/cbs_kernel_resume.so'")
            code=code.replace("'cbs_kernel_query_disk_candidate_20261022.cpp', 'cbs_query_disk_memo_candidate_20261022.hpp', 'cbs_start_signature_cache_v2_20261022.hpp', 'cbs_kernel_query_disk_candidate_20261022.so'", "'native_resume/cbs_kernel_query_disk_candidate_20261022.cpp', 'native_resume/cbs_query_disk_memo_candidate_20261022.hpp', 'native_resume/cbs_start_signature_cache_v2_20261022.hpp', 'native_resume/cbs_kernel_resume.so'")
        out=backend/src.name;out.write_text(code)
        if code!=old:(backend/(src.name+'.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),code.splitlines(True),fromfile=str(src),tofile=str(out))))
        records.append(dict(source=str(src),sha256=sha(src),target=str(out),target_sha256=sha(out),path_only_change=code!=old))
    write(backend/'BUILD_RECEIPT.json',dict(records=records,compiler_command=command,native_library=str(library),native_sha256=sha(library),cache_root=str(HERE/'runtime_cbs_query_cache_resume')))
    print('Safe private backend built:',len(records),'sources',flush=True)
if __name__=='__main__':main()
