"""Execute the already-validated dense runner with the frozen k1 checkpoint.

Only experiment paths and checkpoint identity are substituted; rollout and
logging semantics are inherited byte-for-byte from the preceding dense audit.
"""
from pathlib import Path

ROOT=Path('/home/zhihan/research/Basin_C1')
source_path=ROOT/'diagnostics/gphi_retrained_dense_strict_deadlock_v1/run_dense.py'
text=source_path.read_text()
text=text.replace('HERE = ROOT / "diagnostics/gphi_retrained_dense_strict_deadlock_v1"','HERE = ROOT / "diagnostics/gphi_dagger_k1_diagnostic_v1"\nSOURCE_MANIFEST = ROOT / "diagnostics/gphi_retrained_dense_strict_deadlock_v1/source_manifest.json"')
text=text.replace('CHECKPOINT = ROOT / "diagnostics/gphi_strict_deadlock_coverage_retrain_v1/best_strict_deadlock_coverage_checkpoint.npz"','CHECKPOINT = ROOT / "diagnostics/gphi_dagger_k1_diagnostic_v1/best_dagger_k1_checkpoint.npz"')
text=text.replace('EXPECTED = "340b81d5c4ad2cea7bee16931fa00d095708f5aa6ca6f255e0a7fc5a35873700"','EXPECTED = "83c704f2e1ce0fbe50abd5a0d3e96dd202b4a89256a4ea0e6a954eda0340f70a"')
text=text.replace('source = json.loads((HERE / "source_manifest.json").read_text())','source = json.loads(SOURCE_MANIFEST.read_text())')
if text.count('83c704f2e1ce0fbe50abd5a0d3e96dd202b4a89256a4ea0e6a954eda0340f70a') != 1:
    raise RuntimeError('runner checkpoint substitution failed')
exec(compile(text,str(source_path), 'exec'), {'__name__':'__main__','__file__':str(source_path)})
