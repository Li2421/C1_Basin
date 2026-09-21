"""Publish complete artifacts by rename; checkpoint owns its update history."""
import json
import os
import pickle


def atomic_save(path, value, binary=False):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb' if binary else 'w') as handle:
        if binary:
            pickle.dump(value, handle)
        else:
            handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def restored_history(saved, history_path):
    history = saved.get('history')
    if history is None:
        history = json.loads(history_path.read_text())
    if (len(history) != saved['completed_updates']
            or [r['update'] for r in history] != list(range(len(history)))):
        raise ValueError('resume checkpoint/history disagreement')
    return history
