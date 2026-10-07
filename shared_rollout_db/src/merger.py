from __future__ import annotations
import argparse,json
from pathlib import Path
from .historical_ingest import Ingestor
from .rollout_db import ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',required=True);a=ap.parse_args();n=0
    ing=Ingestor()
    for p in sorted((ROOT/'journals'/a.experiment).glob('*.jsonl')):
        ing.ingest_jsonl(p,experiment_override=a.experiment,journal=True)
        old=str(p.resolve());done=p.with_suffix('.merged');p.rename(done)
        ing.con.execute('UPDATE source_file SET path=? WHERE path=?',(str(done.resolve()),old));ing.con.commit();n+=1
    ing.finish();print(json.dumps({'merged_journals':n}))
if __name__=='__main__':main()
