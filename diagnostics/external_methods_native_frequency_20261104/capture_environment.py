"""Bind numerical dependency versions, including legacy vendored dependencies."""
from native_io import *
import subprocess,argparse
SCRIPT='''import sys,json,importlib.metadata as m
if sys.argv[1]:sys.path.insert(0,sys.argv[1])
names=['numpy','scipy','cvxopt','shapely','cvxpy','clarabel','qpsolvers','jax','jaxlib','torch','gpytorch','matplotlib','PyQt5']
versions={}
for n in names:
 try:versions[n]=m.version(n)
 except m.PackageNotFoundError:versions[n]=None
print(json.dumps(dict(python=sys.version,executable=sys.executable,versions=versions)))
'''
def capture():
    cases=read(D/'ACTIVE_NATIVE_CASES.json');configs=sorted({(c['payload']['python'],c['payload'].get('legacy_root','')) for c in cases});rows={}
    for exe,root in configs:
        deps=str(Path(root)/'inputs/diagnostics/paper_baselines_20261028/deps') if root else ''
        p=subprocess.run([exe,'-B','-c',SCRIPT,deps],text=True,capture_output=True,check=True)
        rows[exe+'|'+root]=json.loads(p.stdout)
    return dict(environments=rows,source_profile_sha256=sha(D/'EXECUTION_SOURCE_PROFILE.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();r=capture()
    if args.check:assert r==read(D/'ENVIRONMENT_BINDING.json')
    else:
        dest=D/'ENVIRONMENT_BINDING.json'
        if dest.exists():assert r==read(dest)
        else:dump(dest,r)
    print('numerical environments',len(r['environments']),'matched' if args.check else 'bound')
