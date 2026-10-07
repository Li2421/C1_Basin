#!/usr/bin/env python3
import json,hashlib,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=sorted(p for p in HERE.rglob('*') if p.is_file() and not any(x in p.parts for x in ('runs','raw','logs','plans','__pycache__')) and p.name!='manifest.json')
x=dict(experiment='ORTHOFLOW3_SHARED_CONSERVATIVE_SUCCESS_SET_FAMILY_V1',created_unix=time.time(),authoritative_basis_sha256='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38',controller_training=False,classification=json.load(open(HERE/'final_decision.json'))['classification'],READY_FOR_MARGIN_LOSS_TRAINING=False,files={str(p.relative_to(HERE)):sha(p) for p in files})
(HERE/'manifest.json').write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps({'files':len(files),'manifest_sha256':sha(HERE/'manifest.json')}))
