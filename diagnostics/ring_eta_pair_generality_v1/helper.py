import os,sys
os.environ['ROUND22_PHASE']='main'
from core import *
import importlib.util
from shared_rollout_db.src import cache_writer
index=int(sys.argv[1]);plan=read(HERE/'RESOURCE_BOOST_PLAN.json');assert sha(HERE/'RESOURCE_BOOST_PLAN.json')==read(HERE/'RESOURCE_BOOST.receipt.json')['plan_sha256'];rows=[c for c in plan['cases'] if c['reservation_owner']==100+index];owned={str((HERE/'main/raw'/f"{c['rollout_uid']}.json").resolve()) for c in rows};original_exists=Path.exists
# Only the assigned helper executes its reservation. Original workers reuse it
# as an operational marker; the sole merger resolves it to the actual record.
def exists(p):
 found=original_exists(p)
 if found and str(p.resolve()) in owned:
  r=read(p)
  if r.get('schema')=='RESERVATION_NOT_EVIDENCE':assert r['reservation_owner']==100+index;return False
 return found
Path.exists=exists
original_append=cache_writer.append_journal;cache_writer.append_journal=lambda rs,*a,**kw:original_append(clean(rs),*a,**kw)
spec=importlib.util.spec_from_file_location('round9_frozen',R9/'run.py');frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen);frozen.execute([{k:v for k,v in c.items() if k!='reservation_owner'} for c in rows],'main',100+index)
