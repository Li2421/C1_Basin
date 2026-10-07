#!/usr/bin/env python3
"""Derive acquisition roles from frozen manifests, never from rollout wrapper labels alone."""
from audit import HERE,read
from collections import defaultdict
def validation_memberships():
 out=defaultdict(set)
 for p in sorted(HERE.glob('targeted_probe_rounds/*/manifest.csv')):
  for r in read(p):
   phase=r.get('phase','')
   if 'retained_validation' not in phase:continue
   out[r['state_id'],r['eta_key']].add(phase)
 return out
