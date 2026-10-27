from common import *
from stages import prepare,capability_select
from dispatch import run
try:
 if not(HERE/'capability_PREFLIGHT.receipt.json').exists():prepare('capability')
 run('capability');mild=capability_select()
 if mild:
  if not(HERE/'capability_mild_PREFLIGHT.receipt.json').exists():prepare('capability_mild')
  run('capability_mild')
 capability_select(True)
 if not(HERE/'main_PREFLIGHT.receipt.json').exists():prepare('main')
 run('main');write(HERE/'ROLLOUTS_COMPLETE.json',dict(utc=utc(),status='COMPLETE'))
except BaseException as exc:
 write(HERE/'ROLLOUT_FATAL.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),traceback=__import__('traceback').format_exc()));raise
