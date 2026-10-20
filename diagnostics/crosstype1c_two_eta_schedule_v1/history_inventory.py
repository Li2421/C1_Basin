"""Read permitted required histories; never enter the mixed-zoo evidence tree."""
from core import *
import csv
def main():
    groups={
    'docs':['generalization_axes_protocol_20261015.md','controller_generalization_audit_20261011.md','controller_generalization_ring_summary_20261014.md','prompt_crosstype1c_selected_two_eta_schedule_20261018.md'],
    'diagnostics/crosstype1b_generator_critic_pipeline_v1':['REPORT.md','HANDOFF.md','GAPS.md','ITERATION_LOG.md','AXIS_TABLE.md','CANDIDATES.json','condition_candidates_v2.py','critic_model.py','action_only_model.py','train_critic.py','select_models.py','select_models_v2.py','select_matched_v2b.py','new_evaluation_choices.py','statistics_pipeline.py','frozen_networks.py','resource_guard.py','FROZEN_PIPELINE.json','NEW_EVAL_BUNDLE.npz','TRAINING_BUNDLE.npz','BASE_PROTOCOL.json','STATE_INPUTS.npz','TRAINING_STATES.json','NEW_EVAL_STATES.json','CONTROLLER_FEATURES.npz','DEVELOPMENT_PROBABILITIES_V2.npz','DEVELOPMENT_CHOICES_V2B.npz'],
    'diagnostics/orthoflow3_generator_critic_v1':['REPORT.md','ARCHITECTURE_AUDIT.md','train_evaluate.py','normalization.json','generator/seed41/checkpoint.msgpack'],
    'diagnostics/orthoflow3_generator_critic_frozen_test_v1':['run_frozen_test.py'],
    'diagnostics/controller_eta_switch_time_v1':['REPORT.md'],
    'diagnostics/ring_eta_pair_generality_v1':['REPORT.md','ETA_PAIR_DRAW.json'],
    'diagnostics/crosstype1_learned_zoo_selector_v1':['ZOO_MANIFEST.csv','features.py','DEVELOPMENT_FEATURES.npz','LOCKED_FEATURES.npz'],
    'diagnostics/blackbox_families_v1':['TUNING_STATES.json','EVAL_STATES.json','episode.py'],
    'diagnostics/crosstype2_mixed_zoo_selector_v1':['REPORT.md','HANDOFF.md','GAPS.md'],
    '.':['AGENTS.md','shared_rollout_db/README.md']}
    inventory=[]
    for folder,names in groups.items():
        for name in names:
            path=ROOT/folder/name;raw=path.read_bytes();row=dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),read=True)
            if path.suffix=='.npz':
                with np.load(path,allow_pickle=False) as z:row['arrays']={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files}
            elif path.suffix=='.json':
                value=json.loads(raw);row['top_level']=list(value)[:30] if isinstance(value,dict) else dict(length=len(value))
            inventory.append(row)
    base=read(D/'crosstype1b_generator_critic_pipeline_v1/BASE_PROTOCOL.json');inputs={}
    for record in base['profiles'].values():
        v=record['payload']
        for key in ('flow_checkpoint_path','adapter_manifest_path'):
            if key in v and Path(v[key]).is_file():inputs[v[key]]=sha(v[key])
        path=Path(v.get('flow_checkpoint_path','/nonexistent')).parent/'NORMALIZATION.json'
        if path.is_file():inputs[str(path)]=sha(path)
    write(HERE/'HISTORY_READ_INVENTORY.json',dict(utc=utc(),files=inventory,controller_inputs=inputs,required_missing=[]))
    report=dict(utc=utc(),reuse=[dict(path=str(D/'crosstype1b_generator_critic_pipeline_v1'),compatible='Same62 controllers,48 training states, first5 exact fixed generator actions, original safety/500/progress references; frozen model only as comparator',forbidden_use='No η13/η10 or anchor labels in new critic/generator training'),dict(path=str(D/'crosstype1_learned_zoo_selector_v1'),compatible='Same24 mechanism features, checkpoints, controller dynamics and reset'),dict(path=str(D/'ring_eta_pair_generality_v1'),compatible='Mechanism background; its fixed K16 pairs are not new generator proposals and cannot label the new pair grid')],gaps=['Three additional fixed generator samples and18 random off-diagonal schedules per training pair lack historical exact labels until preflight','Fresh24 states untouched; only dedup/outcome-free inputs may precede freeze','Historical unrecoverable physical coordinates remain an explicit dedup limitation'],cross_type2_usage='Only three permitted background documents read; no code/model/local labels. DB exact-profile reuse remains permitted.',new_controllers_trained=0,new_probes=0)
    write(HERE/'HISTORY_AUDIT.json',report);seal('HISTORY_AUDIT.receipt.json',[HERE/'HISTORY_READ_INVENTORY.json',HERE/'HISTORY_AUDIT.json',HERE/'history_inventory.py',HERE/'BOOTSTRAP_SOURCES.json'])
    print('Read',len(inventory),'required history files; recorded',len(inputs),'controller inputs')
if __name__=='__main__':main()
