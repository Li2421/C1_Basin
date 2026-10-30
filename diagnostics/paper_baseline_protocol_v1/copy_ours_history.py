"""Copy archived C1 result streams before compatibility/preflight decisions."""
import guard
from guard import HERE,ROOT
import hashlib
import json
import shutil


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    sources=[ROOT/'diagnostics'/name/'candidates.jsonl' for name in
             ('scene_types_oracle_coverage_v3','bottleneck_one_way_admission_v1')]
    sources.append(ROOT/'diagnostics/scene_types_oracle_coverage_v3/SEARCH_STATE.json')
    rows=[]
    for source in sources:
        target=HERE/'inputs'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        before=source.stat()
        expected=sha(source)
        if not target.exists():shutil.copyfile(source,target)
        assert sha(target)==expected,('Historical input copy differs',source)
        after=source.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Historical source is still being written'
        rows.append(dict(source=str(source),copy=str(target.relative_to(HERE)),sha256=expected,bytes=before.st_size))
    value=dict(files=rows,new_rollouts=0,
               scope='Archived labels copied only. Compatibility, RNG/source fingerprints and sampled50-step prefixes still require certification before reuse. No repaired paper labels reused.')
    (HERE/'OURS_HISTORY_COPY.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print('Archived C1 streams copied:',sum(r['bytes']for r in rows),'bytes; no compatibility claim yet.')


if __name__=='__main__':main()
