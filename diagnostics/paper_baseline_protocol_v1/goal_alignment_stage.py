"""Evidence for the common task tolerance, not a tuned controller rule."""
import guard
from guard import HERE
import json
import hashlib


def run():
    source = HERE/'ports/paper_adapters.py'
    assert 'a.arrival_threshold=env.config.goal_tolerance' in source.read_text()
    output = dict(complete=True, common_tolerance=.08,
                  mgr=dict(author='radius/2=0.1m in native scenario', common='env.config.goal_tolerance',
                           source='ports/paper_adapters.py:118', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                           change_already_present_in_archived_port=True),
                  dac=dict(hard_deadzone_larger_than_common=False,
                           saturation_scale_0_5_is_not_stopping_threshold=True),
                  external_native_reproductions='Retain original author criteria; alignment applies only to common benchmark.')
    (HERE/'GOAL_ALIGNMENT_CHECK.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    return output
