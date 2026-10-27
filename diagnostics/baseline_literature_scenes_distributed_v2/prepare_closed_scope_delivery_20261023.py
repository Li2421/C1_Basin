from experiment import *
from atomic_write import write
import ast,difflib
files=[Path(__file__)];changes=[]
def output(old,new,after):
 before=(HERE/old).read_text();ast.parse(after);target=HERE/new;assert not target.exists();target.write_text(after);diff=HERE/(new+'.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new)));files.extend([target,diff]);changes.append(dict(source=old,source_sha256=sha(HERE/old),target=new,target_sha256=sha(target)))
old='final_reports_correction_delivery_20261023.py';s=(HERE/old).read_text()
a='def main():\n';s=s.replace(a,a+" verify_receipt('CORRECTION_CLOSED_SCOPE_DELIVERY_READY_20261023.receipt.json')\n",1)
a=" for gap in (read(HERE/'OPTIONAL_GAPS.json')";b=" notes.append('旧圆形闭合范围固定为用户停止时封存的2044条。修订阶段依指令补齐η0时，相同执行身份可能进入全局缓存；它们归新阶段授权，不扩大旧学习范围，不视为恢复旧批次。DB_POSTFLIGHT另外列出这些跨范围η0缓存命中，旧4100条仍按停止决定列示。')\n"+a;assert a in s;s=s.replace(a,b)
s=s.replace("'final_reports_correction_delivery_20261023.py'","'final_reports_correction_closed_scope_20261023.py'")
output(old,'final_reports_correction_closed_scope_20261023.py',s)
old='optional_delivery_correction_delivery_20261023.py';s=(HERE/old).read_text().replace('postflight_correction_revised_20261021.py','postflight_correction_closed_scope_20261023.py').replace('final_reports_correction_delivery_20261023.py','final_reports_correction_closed_scope_20261023.py');s=s.replace('def main():\n',"def main():\n    verify_receipt('CORRECTION_CLOSED_SCOPE_DELIVERY_READY_20261023.receipt.json')\n",1)
output(old,'optional_delivery_correction_closed_scope_20261023.py',s)
old='correction_after_distributed_fixed_delivery_20261023.py';s=(HERE/old).read_text();pairs=[('statistics_parallel_correction_20261023.py','statistics_parallel_scoped_20261023.py'),('CORRECTION_PARALLEL_NUMERICS_READY_20261023.receipt.json','CORRECTION_SCOPED_NUMERICS_READY_20261023.receipt.json'),('optional_delivery_correction_delivery_20261023.py','optional_delivery_correction_closed_scope_20261023.py')]
for a,b in pairs:assert a in s,a;s=s.replace(a,b)
s=s.replace("wait_ready('CORRECTION_FINAL_DELIVERY_READY_20261023.receipt.json'","wait_ready('CORRECTION_CLOSED_SCOPE_DELIVERY_READY_20261023.receipt.json'")
s=s.replace('def main():\n',"def main():\n verify_receipt('CORRECTION_CLOSED_SCOPE_DELIVERY_READY_20261023.receipt.json')\n",1)
output(old,'correction_after_distributed_fixed_closed_scope_20261023.py',s)
write(HERE/'CLOSED_SCOPE_DELIVERY_SOURCE_DERIVATIONS_20261023.json',dict(utc=utc(),changes=changes,scientific_requests_unchanged=True,math_unchanged=True,change='Apply closed2044 old-circle scope in original-label diagnoses and final DB audit; authorized new eta0 sameUID cache hits belong to new scope,never enlarge old cancelled batch'))
files += [HERE/'CLOSED_SCOPE_DELIVERY_SOURCE_DERIVATIONS_20261023.json',HERE/'CORRECTION_SCOPED_NUMERICS_READY_20261023.receipt.json',HERE/'CORRECTION_FINAL_DELIVERY_READY_20261023.receipt.json']
seal('CORRECTION_CLOSED_SCOPE_DELIVERY_READY_20261023.receipt.json',files,no_old_scope_resumption=True,authorized_eta0_reuse_preserved=True,science_and_numeric_kernel_unchanged=True,only_saved_evidence_scope_and_audit_adapters=True)
print('Closed-scope final delivery ready')
