"""Point historical comparison pages to the declared native-rate primary panel.

Keep all original numeric tables and frozen execution sources unchanged.
"""
from native_io import *
import re
MARK='NATIVE_FREQUENCY_PRIMARY_20261104'
def main():
    status=read(D/'RESULTS.json')['status']
    text='原生频率比较：'+status+'。本页保留此前按场景步长运行的历史结果；DAC、环流等存在控制周期差异，方法排名请以原生频率比较入口为准。'
    changed=[]
    for name in ['c1_v3_new_baseline_reproduction_20261103','c1_v3_mcbf_circulation_paired_20261104','method_comparison_success_time_20261102']:
        root=ROOT/'diagnostics'/name;p=root/'comparison.html'
        banner='<aside id="'+MARK+'" style="padding:16px;border:2px solid #b7791f;background:#fffaf0"><strong>'+text+'</strong> <a href="../'+D.name+'/comparison.html">打开原生频率比较</a></aside>'
        if p.exists():
            original=p.read_text();new=re.sub(r'<aside id="'+MARK+r'".*?</aside>','',original,flags=re.S)
            # Existing generated pages omit an explicit body; insert immediately
            # before their visible title rather than before HTML metadata.
            i=new.find('<h1>');assert i>=0
            new=new[:i]+banner+new[i:];p.write_text(new);changed.append(str(p))
        report=root/'REPORT.txt'
        if report.exists():
            old=report.read_text();old=re.sub(r'^\['+MARK+r'\].*?\[/'+MARK+r'\]\n\n','',old,flags=re.S)
            report.write_text('['+MARK+']\n'+text+'\n原生频率主入口：../'+D.name+'/comparison.html\n[/'+MARK+']\n\n'+old)
        (root/'CURRENT_COMPARISON_NOTICE.txt').write_text(text+'\n原生频率主入口：../'+D.name+'/comparison.html\n完整说明：../'+D.name+'/REPORT.txt\n')
    dump(D/'PUBLISHED_ENTRY_CHECK.json',dict(status=status,changed_pages=changed,numeric_tables_unchanged=True))
    print('updated primary entry notices',len(changed))

if __name__=='__main__':main()
