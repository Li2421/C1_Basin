"""Run the validated fresh-WIDE H8 evaluator with the frozen k1 checkpoint."""
from pathlib import Path
ROOT=Path('/home/zhihan/research/Basin_C1')
source_path=ROOT/'diagnostics/gphi_strict_deadlock_coverage_retrain_v1/run_closed_loop.py'
text=source_path.read_text()
text=text.replace('HERE = ROOT / "diagnostics/gphi_strict_deadlock_coverage_retrain_v1"','HERE = ROOT / "diagnostics/gphi_dagger_k1_diagnostic_v1"')
text=text.replace('DATASET = ROOT / "diagnostics/gphi_training_dataset_strict_deadlock_v1"','DATASET = ROOT / "diagnostics/gphi_training_dataset_dagger_k1_v1"')
text=text.replace('CHECKPOINT = HERE / "best_strict_deadlock_coverage_checkpoint.npz"','CHECKPOINT = HERE / "best_dagger_k1_checkpoint.npz"')
if text.count('best_dagger_k1_checkpoint.npz')!=1: raise RuntimeError('checkpoint substitution failed')
exec(compile(text,str(source_path),'exec'),{'__name__':'__main__','__file__':str(source_path)})
