"""Audit copied historical paper evidence; never import an old execution module."""
import guard
from guard import HERE
import ast
import collections
import datetime
import hashlib
import json
import re
import struct
import zipfile


def read_json(path):
    return json.loads(path.read_text())


def scalar_json(archive, name):
    """Read only a scalar Unicode .npy member, without loading physical arrays."""
    with archive.open(name + '.npy') as stream:
        assert stream.read(6) == b'\x93NUMPY'
        version = stream.read(2)
        width = 2 if version[0] == 1 else 4
        length = struct.unpack('<H' if width == 2 else '<I', stream.read(width))[0]
        header = ast.literal_eval(stream.read(length).decode('latin1'))
        assert header['shape'] == () and not header['fortran_order']
        dtype = header['descr']
        assert dtype.startswith(('<U', '>U')), dtype
        text = stream.read().decode('utf-32-le' if dtype[0] == '<' else 'utf-32-be')
        return json.loads(text.rstrip('\x00'))


def location(record):
    frames = re.findall(r'File "([^"]+)", line (\d+), in ([^\n]+)', record.get('traceback', ''))
    if not frames:
        return dict(file=None, line=None, function=None, accuracy='not_recorded')
    path, line, function = frames[-1]
    return dict(file=path, line=int(line), function=function,
                accuracy='dynamic_compilation_line_not_original_source' if '/paper_method_versions/' in path else 'recorded_source_line')


def assessment(method, key):
    if key.startswith('MethodFailure: qpSWIFT ExitFlag='):
        return dict(status='source_supported_adapter_return_handling_difference',
                    explanation='旧适配层按非零退出码抛异常，阻止作者调用点检查返回解。需恢复作者判断后实跑，不能从退出码推定局部问题无可行解。',
                    evidence=['inputs/diagnostics/scene_types_oracle_coverage_v3/paper_methods/dac_author_si/adapter.py:61',
                              'inputs/diagnostics/paper_baselines_20261028/vendor/dac/method/CBF_DeadlockSolver.py:412-429'])
    if key.startswith('MethodFailure: cvxopt '):
        return dict(status='adapter_status_rejection_requires_call_site_review',
                    explanation='旧适配层对非 optimal 返回直接抛异常；需逐个作者调用点核验返回解处理，不得据此放宽容差或更换求解器。', evidence=[])
    if key.startswith('NumericalUnknown:'):
        return dict(status='solver_status_observed_root_cause_unresolved',
                    explanation='轨迹证明求解器返回该状态；尚不能区分离散状态不可行、模型端口差异和数值求解失败，也不是连续修正族不可能性的证书。', evidence=[])
    return dict(status='native_numeric_exception_requires_source_review',
                explanation='原生求解计算异常，保留栈与受影响局。没有来源支持前不改变公式或数值设置。', evidence=[])


def main():
    manifest = read_json(HERE / 'PAPER_HISTORY_COPY.json')
    assert not manifest['missing']
    groups = {}
    methods = collections.defaultdict(lambda: dict(episodes=0, trace_checked=0, episodes_with_unhandled=0, setup_errors=0, unified_success=0))
    trace_checks = []
    for item in manifest['copied_rollouts']:
        path = HERE / item['raw']
        row = read_json(path)
        method = row['paper_method']
        summary = methods[method]
        summary['episodes'] += 1
        summary['unified_success'] += int(row['success'])
        events = row.get('paper_error_events', [])
        errors = row.get('paper_error_types', {})
        summary['episodes_with_unhandled'] += int(bool(events or row.get('setup_error') or row.get('error')))
        summary['setup_errors'] += int(bool(row.get('setup_error')))
        trace = HERE / item['trace']
        with zipfile.ZipFile(trace) as archive:
            checked_events = scalar_json(archive, 'error_events_json')
            checked_errors = scalar_json(archive, 'error_types_json')
        assert checked_events == events and checked_errors == errors, item['rollout_uid']
        summary['trace_checked'] += 1
        trace_checks.append(dict(rollout_uid=item['rollout_uid'], raw=item['raw'], trace=item['trace'],
                                 raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), error_arrays_equal=True))
        counts = collections.Counter(e['error_key'] for e in events)
        records = dict(errors)
        for label in ('setup_error', 'error'):
            value = row.get(label)
            if value:
                key = label + ': ' + value.get('type', 'unknown') + ': ' + value.get('message', '')
                records[key] = value
                counts[key] += 1
        for key, count in counts.items():
            error = records[key]
            where = location(error)
            group_key = (method, key, where['file'], where['line'], row['type_id'], len(row['initial_positions']))
            if group_key not in groups:
                groups[group_key] = dict(method=method, exception=key, throwing_location=where,
                                         scene_type=row['type_id'], robots=len(row['initial_positions']),
                                         event_count=0, episode_count=0, examples=[], assessment=assessment(method, key))
            group = groups[group_key]
            group['event_count'] += count
            group['episode_count'] += 1
            if len(group['examples']) < 3:
                group['examples'].append(dict(rollout_uid=row['rollout_uid'], raw=item['raw'], trace=item['trace'],
                                              profile=row.get('paper_profile'), purpose=row.get('purpose'),
                                              traceback=error.get('traceback')))
    output = dict(created_at=datetime.datetime.now().astimezone().isoformat(),
                  scope='All copied historical paper labels, deduplicated by rollout UID; includes solo, pilot, probe and joint purposes and multiple profiles.',
                  note='Episode counts within different exception groups overlap. Unified outcomes and first-unhandled outcomes are separate. Dynamically compiled line numbers must not be treated as original formula locations.',
                  new_rollouts=0, methods=dict(methods),
                  groups=sorted(groups.values(), key=lambda x: (x['method'], x['exception'], x['scene_type'], x['robots'])),
                  trace_checks=trace_checks)
    (HERE / 'EXCEPTION_TAXONOMY.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(episodes=sum(m['episodes'] for m in methods.values()),
                          trace_checks=len(trace_checks), groups=len(groups), new_rollouts=0, methods=dict(methods)), ensure_ascii=False))


if __name__ == '__main__':
    main()
