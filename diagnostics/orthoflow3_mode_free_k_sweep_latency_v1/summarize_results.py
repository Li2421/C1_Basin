#!/usr/bin/env python3
"""Build immutable presentation tables from frozen sweep and synchronized timing."""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    sweep = list(csv.DictReader((OUT / 'k_sweep.csv').open()))
    gpu = json.loads((OUT / 'latency_gpu.json').read_text())
    cpu = json.loads((OUT / 'latency_cpu.json').read_text())
    rows = []
    for r in sweep:
        k = r['K']
        g, c = gpu['K'][k], cpu['K'][k]
        rows.append({
            'K': int(k),
            'oracle_B15_states': int(r['oracle_B15_states']),
            'critic_B15_states': int(r['critic_B15_states']),
            'coverage': float(r['coverage']),
            'oracle_critic_gap_states': int(r['oracle_critic_B15_gap_states']),
            'gpu_batched_p50_ms': g['end_to_end']['p50_ms'],
            'gpu_batched_p95_ms': g['end_to_end']['p95_ms'],
            'gpu_batched_p99_ms': g['end_to_end']['p99_ms'],
            'gpu_theoretical_update_hz_p50': g['theoretical_update_hz_from_p50'],
            'gpu_sequential_p50_ms': g['sequential_K_diagnostic']['p50_ms'],
            'cpu_batched_p50_ms': c['end_to_end']['p50_ms'],
            'cpu_batched_p95_ms': c['end_to_end']['p95_ms'],
            'cpu_batched_p99_ms': c['end_to_end']['p99_ms'],
            'cpu_theoretical_update_hz_p50': c['theoretical_update_hz_from_p50'],
            'cpu_sequential_p50_ms': c['sequential_K_diagnostic']['p50_ms'],
            'gpu_hypothetical_loop_with_h0_p50_ms': (
                gpu['state_encoding']['p50_ms'] + gpu['base_safety_step']['p50_ms'] + g['end_to_end']['p50_ms']),
            'cpu_hypothetical_loop_with_h0_p50_ms': (
                cpu['state_encoding']['p50_ms'] + cpu['base_safety_step']['p50_ms'] + c['end_to_end']['p50_ms']),
        })
    write_csv('success_latency_tradeoff.csv', rows)
    component_rows = []
    for device, timing in [('gpu', gpu), ('cpu', cpu)]:
        for component in ('state_encoding', 'base_safety_step'):
            component_rows.append({'device':device,'K':'all','component':component,**timing[component]})
        for k in ('1','2','4','8','16'):
            for component in ('generator','critic','selection','end_to_end','sequential_K_diagnostic'):
                component_rows.append({'device':device,'K':k,'component':component,**timing['K'][k][component]})
    write_csv('latency_components.csv', component_rows)
    audit = json.loads((OUT / 'failure_audit_summary.json').read_text())
    post = json.loads((OUT / 'cache_postflight.json').read_text())['summary']
    result = {
        'classification': 'LARGER_K_WORTH_LATENCY',
        'preferred_K_gpu': 16,
        'preferred_K_cpu_if_latency_constrained': 8,
        'preferred_K_cpu_if_success_prioritized': 16,
        'multimodal_generator_training_justified_now': False,
        'K32_performed': False,
        'K32_reason': 'Would require 51,200 additional matched Q16 continuations; K16 already isolates six remaining coverage failures.',
        'K4_failure_audit': audit,
        'postflight': post,
        'numerical_censoring': json.loads((OUT / 'numerical_censoring.json').read_text()),
        'GPU_latency_source': 'latency_gpu.json',
        'CPU_latency_source': 'latency_cpu.json',
        'success_latency_source': 'success_latency_tradeoff.csv',
    }
    (OUT / 'final_decision.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'decision': result['classification'], 'tradeoff_rows':len(rows)}))


if __name__ == '__main__':
    main()
