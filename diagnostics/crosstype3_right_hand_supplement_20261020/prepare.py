from pathlib import Path
import json,hashlib,shutil
P=Path(__file__).resolve().parent;ROOT=P.parents[1];SRC=ROOT/'diagnostics/crosstype3_unseen_controller_schedule_v1';C2=ROOT/'diagnostics/crosstype2_mixed_zoo_selector_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(P/n).write_text(json.dumps(x,indent=2)+'\n')
files={}
for f in [*SRC.glob('*.py'),*SRC.glob('*.so')]:
 files[str(f)]=sha(f)
 if f.name=='prepare.py':continue
 shutil.copy2(f,P/f.name)
# Local copies isolate all mutable output paths. Nominal controllers, physics,
# safety filter, RNG and event code are copied verbatim.
f=P/'common.py';f.write_text(f.read_text().replace("EXP='exp_crosstype3_unseen_controller_schedule_v1'","EXP='exp_crosstype3_right_hand_supplement_20261020'"))
text='''# Three previously unseen types: fixed right-hand rule supplement\n\nUser authorized temporary reassignment of18 CPU workers from cross-type6, then restoration.\nReuse exact cross-type3 new-state, controller and seed identities:3 types x6 controllers x48 states x2 seeds=1728 requests.\nFixed historical RHR: progress window40 physical steps, progress threshold0.05, tangential speed0.35; safe baseline then final unchanged safety projection. No model fitting, rule tuning, new states or selection.\nCheck exact shared DB identities and execute missing seeds only; unknown is retained and never rerun. Timeout is not deadlock. Existing nine-type RHR evidence is retained without rerun.\nReproduce historical RHR and all three new nominal controller types before evaluation. Only new supplement execution permits RHR around the old nominal source injection; old protected files remain unchanged.\nReport success/collision/timeout/unknown by controller and type, paired with original Safety/full pipeline/NoC on identical states and seeds where available. Bootstrap controller/state clusters with both seeds together.\nPause coordinator before cancelling old workers; retain raw/journals; restore original workers from missing records and resume coordinator even if supplement fails.\n'''
(P/'PREREGISTRATION.md').write_text(text);write('PREREGISTRATION.receipt.json',{'sha256':sha(P/'PREREGISTRATION.md')})
# Remove only the old experiment's prohibition of an arm absent from that protocol.
f=P/'new_execution.py';s=f.read_text();s=s.replace("    assert c.get('rhr_rule') is None,'Continuing right-hand rules are outside this experiment'\n",'');f.write_text(s)
write('SOURCE_COPY_HASHES.json',files)
write('HISTORY_AUDIT.json',{'source_c3':str(SRC),'source_c2':str(C2),'existing_nine_types':'C2 RHR covers2 controllers/type24states2seeds; no rerun requested','three_new_types':'C3 original protocol explicitly excluded RHR; cached original Safety/full/NoC labels retained','compatibility':'reuse exact controller nominal profiles, physical state/RNG namespace,seed,H700,safety config; add exact historical continuing_right_hand identity','gap':'1728 RHR requests before official exact preflight','no_training':True})
print('prepared local source copies',len(files))
