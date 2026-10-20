"""Find exact old 200-step probe prefixes; no task labels and no simulation."""
from common import *
from history_audit import signature


def main():
    p=read(HERE/'HISTORICAL_BASE_PROTOCOL.json');wanted=p['profiles'];states=read(HERE/'PROBE_STATES.json')
    state_keys={signature(s):i for i,s in enumerate(states)}
    sources=[(D/'controller_flow_blend_v1/stage_2/probes','CASES.json','protocol.json'),
             (D/'axis2_controller_v1','probe_CASES.json','probe_protocol.json'),
             (D/'blackbox_families_v1','probe_pilot_deterministic_CASES.json','probe_pilot_deterministic_protocol.json')]
    rows=[];inputs={}
    for folder,casefile,protocolfile in sources:
        protocol=read(folder/protocolfile);cases=read(folder/casefile)
        inputs[str(folder/protocolfile)]=sha(folder/protocolfile);inputs[str(folder/casefile)]=sha(folder/casefile)
        oldstates={s['state_uid']:s for s in protocol['states']}
        rawfolder=folder/'raw' if casefile=='CASES.json' else folder/casefile.replace('_CASES.json','')/'raw'
        for c in cases:
            name=c.get('base_controller_name',c.get('controller_name'))
            if name not in wanted:continue
            oldstate=oldstates[c['state_uid']];key=signature(oldstate)
            if key not in state_keys:continue
            raw=rawfolder/f"{c['rollout_uid']}.json"
            if not raw.exists():continue
            record=read(raw)
            assert record['terminal_labels_read'] is False and not record['jax_enable_x64']
            assert not any(k in record for k in ('success','outcome','timeout','collision'))
            if 'execution_controller_uids' in protocol:
                assert protocol['execution_controller_uids'][name]==wanted[name]['controller_uid']
            else:
                assert protocol['profiles'][name]['payload']['flow_checkpoint_sha256']==wanted[name]['payload']['flow_checkpoint_sha256']
            assert sha(record['trace_path'])==record['trace_sha256']
            rows.append(dict(controller=name,state_index=state_keys[key],arm='F10' if c['eta_index']==10 else 'F13',
                state_signature=key,checkpoint_sha256=wanted[name]['payload']['flow_checkpoint_sha256'],
                raw_path=str(raw),raw_sha256=sha(raw),trace_path=record['trace_path'],trace_sha256=record['trace_sha256'],
                old_physical_steps=record['physical_steps'],complete=record['complete'],
                placement='FF' if wanted[name]['payload'].get('controller_class','Flow')=='Flow' else 'TT'))
    write(HERE/'PROBE_PREFIX_INVENTORY.json',dict(utc=utc(),sources=inputs,prefixes=rows,
          reusable_steps=sum(r['old_physical_steps'] for r in rows),
          task_labels_read=False,new_physical_steps=0,explanation='Recover nominal actions by checkpoint inference on stored states; recover swept geometry from stored segments; continue only missing suffix.'))
    print('Probe prefixes',len(rows),'physical steps reusable',sum(r['old_physical_steps'] for r in rows))

if __name__=='__main__':main()
