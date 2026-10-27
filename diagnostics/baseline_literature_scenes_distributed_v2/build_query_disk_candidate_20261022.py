"""Offline candidate only. Never changes active entrypoints or scientific jobs."""
from experiment import *
from atomic_write import write
import ast,difflib,subprocess


def copy(origin,target,replacements):
    before=(HERE/origin).read_text();after=before
    for a,b in replacements:after=after.replace(a,b)
    if target.endswith('.py'):ast.parse(after)
    p=HERE/target
    assert not p.exists(),p
    p.write_text(after)
    p.with_suffix(p.suffix+'.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=origin,tofile=target)))


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    cpp='cbs_kernel_query_disk_candidate_20261022.cpp';header='cbs_query_disk_memo_candidate_20261022.hpp';library='cbs_kernel_query_disk_candidate_20261022.so'
    old=(HERE/'cbs_kernel_start_signature_v2_candidate_20261022.cpp').read_text()
    new=old.replace('#include "cbs_start_signature_cache_v2_20261022.hpp"','#include "cbs_start_signature_cache_v2_20261022.hpp"\n#include "'+header+'"').replace('start_signature_memo','start_signature_disk_memo')
    new+='\nextern "C" int query_disk_cache_configure(const char* name){return start_signature_disk_memo.disk.configure(name);}\n'
    new+='extern "C" void query_disk_cache_enable(int yes){start_signature_disk_memo.disk.enable(yes!=0);}\n'
    new+='extern "C" void query_disk_cache_stats(uint64_t* out){auto& d=start_signature_disk_memo.disk;out[0]=d.reads;out[1]=d.hits;out[2]=d.collisions;out[3]=d.invalid;out[4]=d.writes;out[5]=d.write_errors;out[6]=d.skipped;}\n'
    new+='extern "C" void query_disk_cache_min_seconds(double seconds){if(seconds>=0 && std::isfinite(seconds))start_signature_disk_memo.minimum_seconds=seconds;}\n'
    p=HERE/cpp;assert not p.exists();p.write_text(new)
    p.with_suffix('.cpp.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='cbs_kernel_start_signature_v2_candidate_20261022.cpp',tofile=cpp)))
    subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(p),'-o',str(HERE/library),'-lcrypto'],check=True)
    tile='tiled_planning_query_disk_candidate_20261022'
    before=(HERE/'tiled_planning_start_signature_v2_20261022.py').read_text();after=before.replace('cbs_kernel_start_signature_v2_candidate_20261022.so',library)
    after+='\n# Separate namespace for each exact candidate implementation; source hashes\n# cannot accidentally reuse artifacts from another prototype.\nQUERY_CACHE_NAMESPACE=hashlib.sha256(b"CBS_QUERY_DISK01"+b"".join((HERE/n).read_bytes() for n in '+repr([cpp,header,'cbs_start_signature_cache_v2_20261022.hpp',library])+')).hexdigest()\n'
    after+='CBS.query_disk_cache_configure.argtypes=[ctypes.c_char_p]\nCBS.query_disk_cache_configure.restype=ctypes.c_int\nQUERY_CACHE_CONFIGURED=bool(CBS.query_disk_cache_configure(QUERY_CACHE_NAMESPACE.encode()))\n'
    ast.parse(after);path=HERE/(tile+'.py');assert not path.exists();path.write_text(after)
    path.with_suffix('.py.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='tiled_planning_start_signature_v2_20261022.py',tofile=path.name)))
    cert='reservation_certificate_query_disk_candidate_20261022';cert2='reservation_certificate_v2_query_disk_candidate_20261022';runtime='runtime_correction_query_disk_candidate_20261022'
    copy('reservation_certificate_start_signature_v2_20261022.py',cert+'.py',[('tiled_planning_start_signature_v2_20261022',tile)])
    copy('reservation_certificate_v2_start_signature_v2_20261022.py',cert2+'.py',[('reservation_certificate_start_signature_v2_20261022',cert)])
    copy('runtime_correction_start_signature_v2_candidate_20261022.py',runtime+'.py',[('reservation_certificate_v2_start_signature_v2_20261022',cert2)])
    names=[cpp,header,library,tile+'.py',cert+'.py',cert2+'.py',runtime+'.py',Path(__file__).name]
    write(HERE/'CBS_QUERY_DISK_CACHE_PLAN_20261022.json',dict(utc=utc(),history=['CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json','CBS_START_SIGNATURE_DEPLOYMENT_20261022.receipt.json','CROSS_ETA_QUERY_REUSE_20261022.json'],gap='Sequential probe saved587/1118native searches; interprocess correctness,overhead and actual benefit still unmeasured',candidate_only=True,production_entry_unchanged=True,new_research_rollouts=0,namespace_slot_data_cap_bytes=65536*16384,cache_policy='Own mutable scratch slots,atomic replacement,full exact key plus SHA256 image check; cache miss/error executes the original search',source_hashes={str(HERE/n):sha(HERE/n) for n in names}))
    print('Offline candidate built; not validated or activated',flush=True)


if __name__=='__main__':main()
