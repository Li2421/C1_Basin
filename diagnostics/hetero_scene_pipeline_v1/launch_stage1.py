from common import *
from design import old_tables
from stages import prepare,capability_select
from dispatch import run
try:
 old_tables();prepare('capability');run('capability');capability_select();prepare('main');run('main')
 write(HERE/'STAGE1_LABELS_COMPLETE.json',dict(utc=utc(),status='COMPLETE'))
except BaseException as exc:
 write(HERE/'RUN_FATAL.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),traceback=__import__('traceback').format_exc()));raise
