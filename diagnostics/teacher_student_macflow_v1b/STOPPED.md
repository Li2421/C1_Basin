Coordinator stopped without retrying or changing any existing source.

```
Traceback (most recent call last):
  File "/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/automate.py", line 34, in <module>
    try:main()
        ^^^^^^
  File "/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/automate.py", line 31, in main
    wait_phase('probe');call('stage.py','mainphase');wait_phase('main');call('postflight.py','main')
                                                     ^^^^^^^^^^^^^^^^^^
  File "/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/automate.py", line 20, in wait_phase
    subprocess.run(['scontrol','update','JobId='+job,'ArrayTaskThrottle='+str(target)],check=True);throttle=target;print('THROTTLE',phase,target,flush=True)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/lib/python3.11/subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['scontrol', 'update', 'JobId=21637', 'ArrayTaskThrottle=20']' returned non-zero exit status 1.

```
