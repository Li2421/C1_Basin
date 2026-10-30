"""Copy every file declared by the archived numerical backend manifest."""
import guard
from guard import HERE,ROOT
import hashlib
import json
import shutil


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=ROOT/'diagnostics/scene_types_oracle_coverage_v3'
    manifest=root/'BACKEND_BUILD.json'
    value=json.loads(manifest.read_text())
    records=[]
    for row in value['sources']:
        source=__import__('pathlib').Path(row['target'])
        assert source.resolve().is_relative_to(root/'generated')
        expected=row['target_sha256']
        assert sha(source)==expected,('Archived backend differs from recorded manifest',source)
        relative=source.relative_to(root)
        for target in (HERE/'inputs/diagnostics/scene_types_oracle_coverage_v3'/relative,HERE/relative):
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(source,target)
            assert sha(target)==expected,('Copied backend differs',target)
        records.append(dict(source=str(source),relative=str(relative),sha256=expected))
    target=HERE/'inputs/diagnostics/scene_types_oracle_coverage_v3/BACKEND_BUILD.json'
    if not target.exists():shutil.copyfile(manifest,target)
    assert sha(target)==sha(manifest)
    canonical=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    output=dict(files=records,manifest_source=str(manifest),manifest_copy=str(target.relative_to(HERE)),
                manifest_sha256=sha(target),canonical_manifest_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                declared_files=len(records),new_rollouts=0,
                note='Completeness of copied source/native inputs, not an execution equivalence certificate.')
    (HERE/'BACKEND_COMPLETENESS.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print('Declared backend files copied and verified:',len(records))


if __name__=='__main__':main()
