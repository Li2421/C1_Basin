"""Reproduce the documented ratio correction using saved bootstrap draws only.

The frozen original audit is retained. A zero-variance bootstrap draw has no
defined proportional reduction; retain it for the absolute difference CI.
No fitting, simulation, or bootstrap resampling takes place here.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def run():
    original = HERE / 'variance_results_v1.json'
    result = json.loads(original.read_text())
    b = np.load(HERE / 'bootstrap.npz')
    for row in result['rows']:
        name = row['interaction']
        common = b[name + '__common']
        independent = b[name + '__independent']
        defined = independent > 1e-20
        ratio = 1 - common[defined] / independent[defined]
        row['conditional_CI95'] = np.quantile(ratio, [.025, .975]).tolist()
        row['undefined_ratio_bootstrap_fraction'] = float(1 - defined.mean())
        row['absolute_variance_reduction_CI95'] = np.quantile(
            independent - common, [.025, .975]).tolist()
    previous = json.loads((HERE / 'variance_results.json').read_text())
    assert result['gate_passed'] == previous['gate_passed']
    for a, c in zip(result['rows'], previous['rows']):
        np.testing.assert_allclose(a['conditional_CI95'], c['conditional_CI95'], atol=1e-12)
    write(HERE / 'variance_results.json', result)
    report = ('LONG-Q NOISE-COUPLING SOURCE SCREEN\n' + result['decision'] + '\n'
              + '\n'.join(str(x) for x in result['rows'] + result['original_noise_signal'])
              + '\nnewrollouts=2048;collisions=0;numericalunknown=7;noQfits.\n'
              + 'Ratio CI excludes undefined zero/zero draws; absolute variance CI retains all draws.\n'
              + 'See ANALYSIS_CORRECTION.json; no new fits, simulations or bootstrap draws.\n')
    (HERE / 'REPORT.txt').write_text(report)
    completion = json.loads((HERE / 'completion.json').read_text())
    completion['report_sha256'] = hashlib.sha256(report.encode()).hexdigest()
    completion['analysis_correction'] = 'correct_ratio_report.py'
    write(HERE / 'completion.json', completion)
    correction = json.loads((HERE / 'ANALYSIS_CORRECTION.json').read_text())
    correction['reproduction'] = 'python -m diagnostics.controller_q_noise_coupling_v1.correct_ratio_report'
    correction['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    correction['undefined_denominator_threshold'] = 1e-20
    write(HERE / 'ANALYSIS_CORRECTION.json', correction)


if __name__ == '__main__':
    run()
