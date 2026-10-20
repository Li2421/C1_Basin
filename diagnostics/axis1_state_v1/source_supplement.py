from common import *
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
paths=sorted(set((FIELD/'source').rglob('*.py'))|set((FIELD/'validation_stage2').rglob('*.py')))
write(HERE/'POC_SOURCE_SUPPLEMENT.json',dict(utc=utc(),files={str(p):sha(p) for p in paths},timing='Supplement after reproduction and main launch; initial required poc/*.py audit exists, additional imported source tree hashes recorded late. Disclose timing in final audit.'))
