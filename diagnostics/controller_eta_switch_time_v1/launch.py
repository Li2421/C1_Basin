"""JSON journal transport normalization; frozen runtime remains unchanged."""
from core import *
import runpy
from shared_rollout_db.src import cache_writer
original_append = cache_writer.append_journal
def append(records, *args, **kwargs):
    return original_append(clean(records), *args, **kwargs)
cache_writer.append_journal = append
if __name__ == '__main__':
    runpy.run_path(str(HERE/'run.py'), run_name='__main__')
