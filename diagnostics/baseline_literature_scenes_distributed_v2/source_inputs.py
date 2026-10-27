"""Read or copy only protected inputs recorded before first v2 reuse."""
from experiment import *

_INDEX = None

def index():
    global _INDEX
    if _INDEX is None:
        start=read(HERE/'INTEGRITY_START.json')
        assert sha(start['manifest'])==start['manifest_sha256']
        _INDEX={}
        with Path(start['manifest']).open() as stream:
            for line in stream:
                r=json.loads(line)
                if 'sha256' in r:_INDEX[r['path']]=r['sha256']
    return _INDEX

def verify(path):
    path=Path(path).resolve()
    expected=index().get(str(path))
    assert expected is not None,('not_in_opening_audit',str(path))
    assert sha(path)==expected,('protected_input_changed',str(path))
    return path

def load(path):
    return read(verify(path))

def copy(path,target,log_name='COPIED_SOURCES.jsonl'):
    path=verify(path);target=Path(target)
    assert target.resolve().is_relative_to(HERE)
    assert not target.exists(),target
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(path.read_bytes())
    assert not (log_name=='COPIED_SOURCES.jsonl' and (HERE/'SOURCE_COPIES.receipt.json').exists()),'Use a separate log for later source copies'
    with (HERE/log_name).open('a') as stream:
        stream.write(canonical(dict(utc=utc(),source=str(path),source_sha256=index()[str(path)],
                                    target=str(target.resolve()),copy_sha256=sha(target)))+'\n')
    return target
