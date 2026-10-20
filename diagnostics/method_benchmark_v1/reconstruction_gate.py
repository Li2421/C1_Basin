from core import *
import shutil
prereg();assert (HERE/'MODEL_METHOD_FREEZE.receipt.json').exists();assert (HERE/'PREDICTIONS_FROZEN.receipt.json').exists()
for path,digest in read(HERE/'MODEL_METHOD_FREEZE.receipt.json')['files'].items():assert sha(path)==digest
old=D/'schedule_phenomenon_consolidated_v1';hashes=read(old/'SOURCE_HASHES.json');assert all(sha(p)==h for p,h in hashes.items()),'Historical input hash changed: STOP'
dest=HERE/'reconstructed';dest.mkdir(exist_ok=True);source=(old/'reconstruct.py').read_text();needle="HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];D=ROOT/'diagnostics';"
assert needle in source
code=source.replace(needle,f"HERE=Path({str(dest)!r});ROOT=Path({str(ROOT)!r});D=ROOT/'diagnostics';")
(HERE/'R21_RECONSTRUCT_SNAPSHOT.py').write_text(source)
exec(compile(code,str(old/'reconstruct.py'),'exec'),{'__file__':str(old/'reconstruct.py'),'__name__':'round26_reconstruction'})
gate=read(dest/'RECONSTRUCTION_GATE.json');baseline=read(old/'RECONSTRUCTION_GATE.json');assert gate['passed'] and gate['checks']==baseline['checks']
arraychecks=[]
for f in dest.glob('R*.npz'):
 a=np.load(f);b=np.load(old/f.name);assert a.files==b.files
 for k in a.files:assert np.array_equal(a[k],b[k],equal_nan=True) if np.issubdtype(a[k].dtype,np.number) else np.array_equal(a[k],b[k])
 arraychecks.append(dict(file=f.name,sha256=sha(f),historical_sha256=sha(old/f.name),byte_identical=sha(f)==sha(old/f.name)))
write(HERE/'RECONSTRUCTION_GATE.json',dict(utc=utc(),passed=True,checks=gate['checks'],array_checks=arraychecks,source_files_checked=len(hashes),all_source_hashes_unchanged=True,code_sha256=sha(old/'reconstruct.py'),after_M3_and_methods_freeze=True));print('R21 reconstruction reproduced',len(gate['checks']),flush=True)
