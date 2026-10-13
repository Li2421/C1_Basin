"""Round 5 immutable artifacts and UTC receipts; no historical writes."""
import hashlib,json,os,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'diagnostics/controller_loco_real_flow_v1'
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def freeze(p,x):
    p=Path(p)
    if p.exists():assert read(p)==x,('immutable',str(p))
    else:write(p,x)
def receipt(p):
    p=Path(p);q=p.with_suffix('.receipt.json')
    if q.exists():assert read(q)['sha256']==sha(p)
    else:write(q,dict(file=str(p),sha256=sha(p),utc=utc(),evidence='Created before this stage reads any new rollout outcome'))
