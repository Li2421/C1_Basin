"""Update our array throttle and verify state despite completed-task warnings."""
import argparse
import re
import subprocess


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('job', type=int);p.add_argument('cap', type=int)
    a = p.parse_args();assert a.job == 13417 and 1 <= a.cap <= 12
    subprocess.run(['scontrol','update',f'JobId={a.job}',f'ArrayTaskThrottle={a.cap}'],
                   text=True, capture_output=True)
    r = subprocess.run(['squeue','-h','-j',str(a.job),'-t','PENDING','-o','%i'],
                       text=True, capture_output=True, check=True)
    caps = [int(v) for v in re.findall(r'%(\d+)',r.stdout)]
    assert all(v == a.cap for v in caps), r.stdout
    print('Verified pending array cap', a.cap, '(or no pending array tasks)')
