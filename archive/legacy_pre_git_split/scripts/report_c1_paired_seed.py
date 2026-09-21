"""Compare completed C1 and rerun baselines under an explicit paired protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.diagnostics.stalled_outcomes import OUTCOMES, PROTOCOL, classify_timeout_trace


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--c1-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--expected-n', type=int, default=200)
    parser.add_argument('--expected-max-steps', type=int, default=850)
    args = parser.parse_args()
    for folder in (args.baseline_dir, args.c1_dir):
        if not (folder/'complete.json').exists():
            raise ValueError(f'incomplete evaluation: {folder}')
    base, c1 = (read(folder/'config.json') for folder in (args.baseline_dir, args.c1_dir))
    for key in ('environment', 'cbf', 'seed', 'split', 'n_rollouts'):
        if base[key] != c1[key]:
            raise ValueError(f'unmatched evaluation {key}')
    n, horizon = args.expected_n, args.expected_max_steps
    if (n, horizon) not in ((200,850), (25,1200)):
        raise ValueError('unsupported comparison protocol')
    if base['n_rollouts'] != n or base['seed'] != 42 or base['environment']['max_steps'] != horizon:
        raise ValueError('evaluation differs from requested comparison protocol')
    if (base['checkpoint_sha256'] != c1['baseline_checkpoint_sha256']
            or base['initial_suite']['sha256'] != c1['initial_states_sha256']):
        raise ValueError('unmatched checkpoint or initial suite')
    residual = Path(c1['residual'])
    if hashlib.sha256(residual.read_bytes()).hexdigest() != c1['residual_sha256']:
        raise ValueError('evaluated residual has changed')
    saved = pickle.loads(residual.read_bytes())
    selected_feasible = bool(saved.get('selection', {}).get('constraint_satisfied', False))
    completed = saved.get('completed_updates', saved.get('selection', {}).get('update', 'unknown'))
    results, rows, nominal_errors = {}, {}, []
    for name, folder in (('MAC-only', args.baseline_dir/'mac_only'),
                         ('Safety', args.baseline_dir/'mac_cbf'), ('C1', args.c1_dir)):
        summaries = read(folder/'summary.json')['rollouts']
        by_id = {r['rollout_id']:r for r in summaries}
        if len(summaries) != n or sorted(by_id) != list(range(n)):
            raise ValueError('each arm must contain all requested distinct rollout IDs')
        entries = []
        for rid in range(n):
            summary = by_id[rid]
            if summary['initial_positions'] != base['initial_positions'][rid]:
                raise ValueError(f'unpaired initial state in {name}, rollout {rid}')
            with np.load(folder/f'rollout_{rid:04d}.npz', allow_pickle=False) as trace:
                label, details = classify_timeout_trace(trace, summary['outcome'], base['environment']['dt'])
                if name == 'C1':
                    with np.load(args.baseline_dir/'mac_only'/f'rollout_{rid:04d}.npz', allow_pickle=False) as baseline_trace:
                        nominal_errors.append(float(np.max(np.abs(
                            trace['c1_nominal'][0].reshape(2,2)-baseline_trace['u_nom'][0]))))
            entries.append(dict(rollout_id=rid, outcome=label, original_outcome=summary['outcome'],
                                episode_steps=summary['episode_steps'], **details))
        rows[name] = entries
        results[name] = dict(n=n, counts={label:sum(r['outcome']==label for r in entries) for label in OUTCOMES},
                             min_pairwise_h=min(r['min_pairwise_h'] for r in summaries),
                             min_wall_h=min(r['min_wall_h'] for r in summaries))
    comparisons = {}
    for name in ('MAC-only', 'Safety'):
        gained = [i for i in range(n) if rows[name][i]['outcome']!='success' and rows['C1'][i]['outcome']=='success']
        lost = [i for i in range(n) if rows[name][i]['outcome']=='success' and rows['C1'][i]['outcome']!='success']
        common = [i for i in range(n) if rows[name][i]['outcome']==rows['C1'][i]['outcome']=='success']
        comparisons[name] = dict(gained_success_ids=gained, lost_success_ids=lost,
            net_success_change=len(gained)-len(lost), common_success_count=len(common),
            mean_common_success_time_delta_seconds=float(np.mean([
                (rows['C1'][i]['episode_steps']-rows[name][i]['episode_steps'])*base['environment']['dt']
                for i in common])) if common else None)
    # Existing frozen-sampler identity regression uses absolute tolerance 2e-6.
    # Common random inputs do not imply bitwise equality of separately compiled
    # float32 samplers. Preserve the observed numerical discrepancy explicitly.
    if max(nominal_errors) > 2e-6:
        raise ValueError('initial nominal sampler discrepancy exceeds existing identity tolerance')
    report = dict(protocol=PROTOCOL, scope=f'single frozen baseline seed, {n} paired cases per arm',
                  initial_nominal_max_abs_error=max(nominal_errors), initial_nominal_identity_atol=2e-6,
                  baseline_checkpoint_sha256=base['checkpoint_sha256'], residual_sha256=c1['residual_sha256'],
                  suite_sha256=c1['initial_states_sha256'], flow_seed=42, max_steps=horizon,
                  c1_training=saved['metadata']['training'], c1_risk_version=saved['metadata']['risk_version'],
                  checkpoint_kind=('selected validation-feasible' if selected_feasible else
                                   'final residual; not certified validation-feasible'),
                  results=results, paired_c1_vs=comparisons, rows=rows)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('report output must be empty')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# C1 / MAC-only / Safety：同条件单 seed 对比', '',
        f'比较三个已完成的运行目录；同一 seed-0 冻结 Flow-BC、固定宽初态套件前 {n} 个初态、Flow seed=42、{horizon} 步及原安全设置。',
        f'C1 检查点来自第 {completed} 次更新；'+
        ('已按独立验证约束可行性选出。' if selected_feasible else '这是终点迭代模型，不是经可行性选择的模型。')+
        f'此处每个方法 {n} 回合，不是双 seed 400 回合结果。', '',
        '| 方法 | 成功 | 墙碰撞 | 智能体碰撞 | 安全死锁 | 停滞死锁 | 其他超时 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for name, result in results.items():
        lines.append('| '+name+' | '+' | '.join(f'{result["counts"][label]} ({100*result["counts"][label]/n:.1f}%)' for label in OUTCOMES)+' |')
    lines += ['', '六类结果复用原基线的超时分类器；首终止事件标签另行保留。', '']
    lines.append(f'全部 {n} 个初态的首步 nominal 对照最大绝对误差为 {max(nominal_errors):.9g}，在既有 sampler 一致性测试的 2e-6 容差内；不声称逐位一致。')
    for name, pair in comparisons.items():
        lines.append(f'C1 相对 {name}：新增成功 {len(pair["gained_success_ids"])} 例，丢失成功 {len(pair["lost_success_ids"])} 例，净变化 {pair["net_success_change"]:+d} 例。')
    lines += ['', '逐例配对、改善/退化 ID、安全裕量及共同成功案例的耗时差见 comparison.json。',
              '本次宽初态测试已用于查看结果；后续据此调整方案属于自适应开发，最终泛化确认需额外未触碰测试。', '']
    (args.out_dir/'REPORT_ZH.md').write_text('\n'.join(lines))
    print(json.dumps(dict(results=results, paired_c1_vs=comparisons), indent=2))


if __name__ == '__main__':
    main()
