"""Validate saved-evidence diagnostics; no simulation, no controller calls."""
import bootstrap
from experiment import *
from swap_diagnostics import summary
import numpy as np

def main():
    selected={}
    for p in (HERE/'capability/raw').glob('*.json'):
        r=read(p);key=(r['case']['state']['scene'],r['outcome'])
        if key in selected:continue
        assert sha(r['trace_path'])==r['trace_sha256']
        with np.load(r['trace_path']) as z:d=summary(r,z)
        assert len(d['grid'])==9 and len(d['local'])==1
        if r['success']:assert d['local']==['arrived']
        if r['collision']:assert d['local']==['collision']
        selected[key]=dict(raw_path=str(p),raw_sha256=sha(p),trace_sha256=r['trace_sha256'],local=d['local'])
        if len(selected)==5:break
    assert any(k[1]=='timeout' for k in selected) and any(k[1]=='collision' for k in selected)
    write(HERE/'SAVED_DIAGNOSTIC_VALIDATION.json',dict(passed=True,source='Capability saved evidence only',new_rollouts=0,controller_calls=0,cases=list(selected.values())))
    print('Saved diagnostic validation passed',list(selected),flush=True)
if __name__=='__main__':main()
