from core import *
p=HERE/'submit.py';s=p.read_text();s=s.replace("slots=min(18,len(read(HERE/f'{phase}_MISSING_CASES.json')))","slots=min(18,len(read(HERE/f'{phase}_MISSING_CASES.json')));throttle=min(slots,max(1,18-others))")
s=s.replace(" if others and not dependency:slots=min(slots,18-others)\n","")
s=s.replace("f'0-{slots-1}%{slots}'","f'0-{slots-1}%{throttle}'")
p.write_text(s)
p=HERE/'resource_monitor.py';s=p.read_text();s=s.replace(" paused&=groups;needed=max(0,allocated-18)"," external=sum(int(s[3]) for s in running if not s[1].startswith('r35_'));budget=max(1,18-external)\n for base in {s[0].split('_')[0] for s in ownjobs}:\n  subprocess.run(['scontrol','update','JobId='+base,'ArrayTaskThrottle='+str(budget)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n paused&=groups;needed=max(0,allocated-18)")
p.write_text(s)
