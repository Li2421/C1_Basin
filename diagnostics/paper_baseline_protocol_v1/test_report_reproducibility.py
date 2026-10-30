"""Meaningful statistics regression: hash randomization cannot change seeded CI/p."""
import json,sys
from pathlib import Path
import reporting as report


def main():
    report.read=lambda name:dict(methods={'dac_author_si':dict(reproducible=True),
        'mgr_author_unicycle_tracking':dict(reproducible=True),'chen_impc_ob_si_tracking':dict(reproducible=False,reason='Synthetic exclusion')})
    rows=[]
    for scene in range(24):
        for state in range(8,24):
            for method,count in [(report.LEARNED,6),('dac_author_si',1),('mgr_author_unicycle_tracking',1)]:
                for nominal in range(count):
                    success=int((scene*7+state*3+nominal*5)%11<7)if count==6 else int((scene+state*2)%5<3)if method=='dac_author_si'else int((scene*3+state)%7<4)
                    rows.append(dict(benchmark='A',part='test',method=method,variant='frozen'if count==6 else'global_tuned',
                        scene=f's{scene:02d}',category=str(scene%4),state=state,nominal=str(nominal),
                        result=dict(success=success,collision=0,timeout=1-success,numerical_failure=0,completion_time=1. if success else None)))
    value=report.comparisons(rows)
    Path(sys.argv[1]).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')


if __name__=='__main__':main()
