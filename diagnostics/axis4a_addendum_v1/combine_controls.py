from common import *
from registry import register
p=dict(states=[],profiles={},scenarios={},eta=None);rows=[];ids=set()
for stage in ['A1','A2']:
 q=read(HERE/f'{stage}_protocol.json');p['eta']=q['eta'];p['profiles'].update(q['profiles']);p['scenarios'].update(q['scenarios'])
 for s in q['states']:
  if s['state_uid'] not in ids:p['states'].append(s);ids.add(s['state_uid'])
 rows.extend(read(HERE/f'{stage}_CASES.json'))
# Interleave levels within controller to reduce late work and keep cache reuse.
names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];rows.sort(key=lambda c:(names.index(c['base_controller_name']),c['state_index'],c['r'],c['arm']))
register('A',p,rows,{v['controller_uid']:v for v in p['profiles'].values()})
