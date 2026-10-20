import os,sys
os.environ['ROUND22_PHASE']=sys.argv[1]
from core import *
import importlib.util
from shared_rollout_db.src import cache_writer
original=cache_writer.append_journal
cache_writer.append_journal=lambda records,*a,**kw:original(clean(records),*a,**kw)
spec=importlib.util.spec_from_file_location('round9_frozen',R9/'run.py');frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
if __name__=='__main__':
 phase=sys.argv[1];index=int(sys.argv[2]);workers=int(sys.argv[3]);rows=read(HERE/f'{phase}_MISSING_CASES.json');frozen.execute(rows[index::workers],phase,index)
