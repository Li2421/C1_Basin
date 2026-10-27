"""Finite no-training test: capability -> fixed25 labels -> statistics -> audit."""
from common import *
from stages import prepare,capability_select
from dispatch import run
import subprocess

def main():
    assert read(HERE/'REPLAY_CHECK.json')['passed']
    if not(HERE/'capability_PREFLIGHT.receipt.json').exists():prepare('capability')
    run('capability');capability_select()
    if not(HERE/'main_PREFLIGHT.receipt.json').exists():prepare('main')
    run('main')
    while not(HERE/'STATISTICS_FROZEN.receipt.json').exists():
        (HERE/'STATUS.md').write_text(f'# 当前状态\n\n{utc()}：全部能力门与主标签已完成；等待按预定口径封存的统计程序就绪。无追加rollout，无训练。\n');time.sleep(10)
    subprocess.run([sys.executable,'-B',str(HERE/'statistics_test.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B',str(HERE/'deliver.py')],cwd=ROOT,check=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(HERE/'RUN_FATAL.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),traceback=__import__('traceback').format_exc()));raise
