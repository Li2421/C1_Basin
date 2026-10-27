"""Copy the transitive runtime sources; preserve every source SHA and diff."""
from audit import HERE,V2,sha,write
import ast
import json

def main():
    opening=json.loads((HERE/'INTEGRITY_START.json').read_text())['protected']
    pending=['dynamics_spatial_mask_20261022','dynamics_diagnostics'];seen=set();records=[]
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);src=V2/(name+'.py')
        if not src.exists():continue
        assert opening[str(src.resolve())]==sha(src),src
        code=src.read_text()
        for node in ast.walk(ast.parse(code)):
            names=[node.module] if isinstance(node,ast.ImportFrom) else [n.name for n in node.names] if isinstance(node,ast.Import) else []
            pending.extend(n.split('.')[0] for n in names if n and (V2/(n.split('.')[0]+'.py')).exists())
        dst=HERE/src.name
        assert not dst.exists(),dst
        dst.write_text(code)
        records.append(dict(source=str(src.resolve()),source_sha256=sha(src),target=str(dst),copy_sha256=sha(dst)))
    # Native sources are built only under this new output boundary.
    for src in list(V2.glob('*.cpp'))+list(V2.glob('*.hpp')):
        dst=HERE/src.name
        if not dst.exists():dst.write_bytes(src.read_bytes())
        records.append(dict(source=str(src.resolve()),source_sha256=sha(src),target=str(dst),copy_sha256=sha(dst)))
    for name in ['CIRCLE_SPEED.json','CONTROLLER_MANIFEST.json','FIXED_ETA.json',*[f'FLOW_STATES_{s}.json' for s in ('lit_narrow_passage','lit_crowded_random','lit_circle_five','lit_doorway')]]:
        src=V2/name;dst=HERE/name;b=src.read_bytes();dst.write_bytes(b)
        records.append(dict(source=str(src.resolve()),source_sha256=sha(src),target=str(dst),copy_sha256=sha(dst)))
    # Historical speed reader expects a path-keyed receipt. The value is copied
    # exactly; only the receipt's location changes.
    write(HERE/'CIRCLE_SPEED.receipt.json',dict(files={str(HERE/'CIRCLE_SPEED.json'):sha(HERE/'CIRCLE_SPEED.json')},source_receipt_sha256=sha(V2/'CIRCLE_SPEED.receipt.json')))
    write(HERE/'COPIED_SOURCES.json',records)
    print('Copied',len(records),'sources/configs; no historical entrypoint executed.')
if __name__=='__main__':main()
