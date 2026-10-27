"""Keep native statistics numerical code; avoid parsing unused geometry census headers."""
from experiment import *
from atomic_write import write
import ast,difflib
source=HERE/'statistics.py';target=HERE/'statistics_header_reader_20261023.py';before=source.read_text()
a='from result_inventory import find_raw,qualification';b=a+'\nfrom reclassify_instances_20261021 import small_diagnosis';assert a in before;after=before.replace(a,b)
a='diag=read(diag_path)';b='diag=small_diagnosis(diag_path.read_bytes())';assert a in after;after=after.replace(a,b)
assert after.replace(b,a).replace('from reclassify_instances_20261021 import small_diagnosis\n','')==before
ast.parse(after);target.write_text(after);diff=HERE/(target.name+'.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=source.name,tofile=target.name)))
write(HERE/'PARALLEL_STATISTICS_NATIVE_KERNEL_SOURCE_20261023.json',dict(utc=utc(),source=source.name,source_sha256=sha(source),target=target.name,target_sha256=sha(target),numeric_source_unchanged=True,change='Only parse diagnostic fields needed by collect; retain full-file SHA256, original records and source paths',new_rollouts=0))
