"""Preserve pre-alignment author safety constants as a separate reference row."""
from core import *


def request_variant(request):
    if request['controller_payload']['paper_method'] != 'mgr_author_unicycle_tracking':
        return request
    # Revision1 already restored author solver returns and exposed C_R, but
    # predates the physical SAFE_RATIO mapping. Its immutable adapter therefore
    # keeps the author's1.1ratio while retaining the common plant/task mapping.
    folder=HERE/'execution_revisions/v001'
    manifest=read(folder/'MANIFEST.json')
    relative='paper_methods/mgr_author_unicycle_tracking/adapter.py'
    adapter=folder/'source'/relative
    assert sha(adapter)==manifest['files'][relative]
    assert '_unaligned_make' not in adapter.read_text()
    request['controller_payload'].update(paper_method_adapter=str(adapter),
        archived_adapter_sha256=sha(adapter),reference='Author1.1safety-ratio before physical alignment; common robot geometry and0.08task unchanged.')
    request['controller_uid']=db.uid('ctl',dict(scenario=request['scenario_uid'],payload=request['controller_payload']))
    request['rollout_uid']=db.uid('roll',dict(state=request['state_uid'],eta=request['eta_uid'],controller=request['controller_uid'],seed=request['seed_key']))
    return request
