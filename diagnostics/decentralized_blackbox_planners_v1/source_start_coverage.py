from common import *
G=guard()
def main():
 begin=read(HERE/'INTEGRITY_BEGIN.json')['files'];early=read(HERE/'imports_tests.json');covered=dict(begin,**early['imported_sources']);loaded={}
 for f in sorted(HERE.glob('imports_*.json')):
  loaded.update(read(f).get('imported_sources',{}))
 research={p:h for p,h in loaded.items() if p.startswith('/home/zhihan/research/') and '/.venv-' not in p and not Path(p).is_relative_to(HERE)};missing=[p for p in research if p not in covered];mismatch=[p for p,h in research.items() if p in covered and covered[p]!=h]
 write(HERE/'SOURCE_START_COVERAGE.json',dict(utc=utc(),passed=not missing and not mismatch,research_imports=research,missing_start_hash=missing,hash_mismatch=mismatch,initial_full_snapshot_sha256=sha(HERE/'INTEGRITY_BEGIN.json'),pre_rollout_import_inventory_sha256=sha(HERE/'imports_tests.json'),pre_rollout_inventory_utc=early['utc'],scope='All imported existing research Python sources; new local copies separately linked by provenance. Third-party libraries separately inventoried and final-hash-checked, with no package installation or update.',guard=guard()));assert not missing and not mismatch
if __name__=='__main__':main()
