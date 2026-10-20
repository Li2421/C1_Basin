from core import *
prereg();groups={};profiles={};states={}
for tag,folder in [('SOURCE','ring_controller_population_v1'),('T1','ring_independent_flow_confirmation_v2'),('T2','controller_universal_eta_schedule_v1'),('T3','controller_new_checkpoint_confirmation_v1/fp32_n8'),('T4','ring_critic_new_recipe_confirmation_v1'),('T5','ring_eta2_gap_v1')]:
 p=D/folder;q=read(p/('FINAL_BASE_PROFILES.json' if tag in ['SOURCE','T4','T5'] else 'protocol.json'))
 if tag=='SOURCE':names=q['eligible_controller_order'];pro=q['profiles']
 elif tag=='T4':names=q['N2'];pro=q['profiles']
 elif tag=='T5':names=q['test_controller_order'];pro=q['profiles']
 else:names=q['controller_order'];pro=q['base_profiles']
 if tag=='T4' and not isinstance(names,list):names=list(names)
 groups[tag]=names
 for n in names:
  v=pro[n]['payload'];profiles[n]=v;assert sha(v['flow_checkpoint_path'])==v['flow_checkpoint_sha256']
 states[tag]=read(p/('TEST_STATES.json' if tag=='T5' else 'FRESH_STATES.json'))
common=read(D/'controller_loco_real_flow_v1/protocol.json')['states'][:128]
write(HERE/'INPUT_IDENTITIES.json',dict(groups=groups,profiles=profiles,states=states,common_states=common,test_labels_read=False))
write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),reused=['R21 reconstruction code and exact source label arrays after M3 freeze','R16 A0/A3 checkpoints and source-only transforms','R15 frozen R5/R6','R17/R18 frozen R5','R4 compatible static R3'],gaps=['R3 on new checkpoints','R3-only R15 A3 training','uniform half-state method and adaptation comparison'],no_rollout_or_probe=True))
print({k:len(v) for k,v in groups.items()},'unique',len(profiles))
