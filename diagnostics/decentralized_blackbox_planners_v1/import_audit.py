from runtime_new import *
import critic_fit,metrics,stats
import shared_rollout_db.src.merger
from audit_utils import source_inventory
G=guard()
source_inventory('IMPORTED_SOURCE_BEGIN.json')
# Vendor provenance includes the intermediate protected copy and original source.
import difflib
records=[]
for path in sorted((HERE/'vendor').rglob('*.py')):
 rel=path.relative_to(HERE/'vendor');intermediate=D/'teacher_student_macflow_v1f/vendor'/rel;original=FIELD/'source'/rel
 records.append(dict(copy=str(path),copy_sha256=sha(path),intermediate_source=str(intermediate),intermediate_sha256=sha(intermediate),original_source=str(original),original_sha256=sha(original) if original.exists() else None,diff_from_intermediate=''.join(difflib.unified_diff(intermediate.read_text().splitlines(True),path.read_text().splitlines(True))),diff_from_original=''.join(difflib.unified_diff(original.read_text().splitlines(True),path.read_text().splitlines(True))) if original.exists() else None))
write(HERE/'VENDOR_PROVENANCE.json',records)
# Establish exact incompatibility of round33 fresh states before deciding on new evaluation requests.
a=read(D/'axis2_controller_v1/main_protocol.json')['states'];b=read(HERE/'EVAL_STATES.json');matches=[(x['uid'],y['uid']) for x in a for y in b if x['physical']==y['physical'] and x['uid']==y['uid'] and x['rng_namespace']==y['rng_namespace']]
write(HERE/'HISTORY_COMPATIBILITY_ADDENDUM.json',dict(utc=utc(),round33_states=32,round28_states=32,physical_and_rng_matches=matches,nominal_only_reuse=0,necessity='Round33 used fresh round33_axis2_eval states and different RNG aliases; no exact matching nominal requests on round28 evaluation states. Different controller encoders alone are not a reason to rerun.',no_new_training=True,guard=guard()))
