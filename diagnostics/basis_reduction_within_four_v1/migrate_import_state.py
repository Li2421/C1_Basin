"""Retarget saved queries after a certified byte-identical module relocation."""
from core import *
from acceleration import accelerated_case

def migrate(old):
 if not old.exists():return
 data=read(old);entries=read(HERE/'SCENES.json')['entries'];scenes={(e['type_id'],e['level']):e['scene']for e in entries};ctls={c['type_code']:c for c in controllers()};pool=next(c for c in read(HERE/'COMBINATIONS.json')['combinations']if c['code']==data['code'])['candidate_pool'];bridge=read(HERE/'REVISION_EQUIVALENT.json')['records'];manifest=read(HERE/'REUSE_MANIFEST.json');passages=read(HERE/'PASSAGES.json')['by_scene']
 def request(identity,index):
  t,l,ctl,si=identity;return accelerated_case(scenes[(t,l)],ctls[ctl],si,pool[index-1],purpose='basis_reduction_screen')
 for g in data['groups']:
  for q in g['queries']:
   c=request(g['identity'],q['index']);assert c['rollout_uid']in bridge or not passages[g['scene']]['passages'],('No proven relocation equivalence',g['identity'],q['index'])
   q['query_uid']=c['rollout_uid'];q['result']=dict(q['result'],logical_query_uid=c['rollout_uid'])
   manifest['records'][c['rollout_uid']]=dict(target_query_uid=c['rollout_uid'],source_rollout_uid=q['result']['rollout_uid'],kind='import_relocation_preserved_query',certificate='IMPORT_RELOCATION_CERT.json',prior_query_preserved_in=str(old))
  if g['success']:g['solution']=g['queries'][g['first_success']-1]
 if data['pending']:
  for item in data['pending']['frontier']:item['query_uid']=request(item['identity'],item['index'])['rollout_uid']
 dump(HERE/old.name,data);dump(HERE/'REUSE_MANIFEST.json',manifest)
 print('Preserved',data['code'],'frontier and',sum(g['searched']for g in data['groups']),'logical query results; no new trajectory generated')

def main():
 folder=HERE/'inputs/utils_relocation'
 if not folder.exists():folder=HERE/'inputs/import_relocation'
 for path in sorted(folder.glob('SEARCH_*.json')):
  if path.name!='SEARCH_four_8_screen.json':migrate(path)

if __name__=='__main__':main()
