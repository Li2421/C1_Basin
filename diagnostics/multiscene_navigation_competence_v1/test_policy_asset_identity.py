from common import *
from collect39 import policy_assets
from shared_rollout_db.src.rollout_db import uid
import tempfile
with tempfile.TemporaryDirectory(dir=HERE/'tmp') as temp:
 p=Path(temp);(p/'best.msgpack').write_bytes(b'unit fixture, not an actual policy');(p/'NORMALIZATION.json').write_text('{"obs_mean": [0]}');(p/'CONFIG.json').write_text('{"observation_dimensions":39}')
 first=policy_assets(p/'best.msgpack');old=uid('ctl',first)
 (p/'NORMALIZATION.json').write_text('{"obs_mean": [1]}');second=policy_assets(p/'best.msgpack');assert uid('ctl',second)!=old
 assert first['checkpoint_sha256']==second['checkpoint_sha256'];assert first['normalization_sha256']!=second['normalization_sha256']
 before=uid('ctl',second);(p/'CONFIG.json').write_text('{"observation_dimensions":39,"unit_fixture":true}');assert uid('ctl',policy_assets(p/'best.msgpack'))!=before
 assert all(v is None for v in policy_assets(None).values())
write(HERE/'POLICY_ASSET_IDENTITY_TEST.json',dict(passed=True,normalization_change_changes_controller_uid=True,config_change_changes_controller_uid=True,checkpoint_same_does_not_imply_same_controller=True,unit_fixture_only=True))
print('normalization/config exact identities passed')
