"""Authoritative full controller-type names from the previous public manifest."""
from common import *
import csv
with (D/'crosstype3_unseen_controller_schedule_v1/ZOO_MANIFEST.csv').open() as f:
    TYPE_NAMES={r['type_code']:r['type'] for r in csv.DictReader(f)}
def full_type(code):return TYPE_NAMES.get(str(code),str(code))
