"""Static report-only correction: retain required source audit as first paragraph."""
from experiment import *
from registry import verify_receipt
import ast


def main():
    os.sched_setaffinity(0, {read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    verify_receipt('FINAL_DELIVERY_CLARITY_20261022.receipt.json')
    proof = 'FINAL_DELIVERY_AUDIT_ORDER_20261022.receipt.json'
    assert not (HERE/proof).exists()
    pairs = [
        ('final_reports_delivery_clarity_20261022.py', 'final_reports_audit_order_20261022.py'),
        ('optional_delivery_final_clarity_20261022.py', 'optional_delivery_audit_order_20261022.py'),
    ]
    copied = []
    for old, new in pairs:
        source = HERE/old
        target = HERE/new
        assert not target.exists()
        text = source.read_text()
        if old.startswith('final_reports'):
            needle = ' paragraphs.insert(0,report_lead(results,cell,delta))'
            replacement = ' paragraphs.insert(1,report_lead(results,cell,delta))'
            assert text.count(needle) == 1
            text = text.replace(needle, replacement)
            text = text.replace(old, new)
            anchor = " verify_receipt('FINAL_DELIVERY_CLARITY_20261022.receipt.json')"
            assert text.count(anchor) == 1
            text = text.replace(anchor, anchor + "\n verify_receipt('" + proof + "')")
        else:
            needle = "action('final_report','final_reports_delivery_clarity_20261022.py')"
            assert text.count(needle) == 1
            text = text.replace(needle, "action('final_report','final_reports_audit_order_20261022.py')")
            anchor = "    verify_receipt('FINAL_DELIVERY_CLARITY_20261022.receipt.json')"
            assert text.count(anchor) == 1
            text = text.replace(anchor, anchor + "\n    verify_receipt('" + proof + "')")
        ast.parse(text)
        target.write_text(text)
        copied.append(dict(source=old, source_sha256=sha(source), target=new, target_sha256=sha(target)))
    write(HERE/'FINAL_DELIVERY_AUDIT_ORDER_20261022.json', dict(
        utc=utc(), copies=copied, reason='Original v2 requires protected-source integrity audit in REPORT first paragraph; primary fixed comparison follows immediately',
        science_changes=False, statistics_changes=False, request_changes=False,
        read_only_history=['FINAL_DELIVERY_CLARITY_20261022.receipt.json', 'MIGRATION_NUMBERS.receipt.json']))
    seal(proof, [Path(__file__), *[HERE/new for _, new in pairs], HERE/'FINAL_DELIVERY_AUDIT_ORDER_20261022.json', HERE/'FINAL_DELIVERY_CLARITY_20261022.receipt.json'], report_only=True, new_rollouts=0, new_bootstrap_draws=0)
    print('Prepared static report-only variant; not activated', flush=True)


if __name__ == '__main__':
    main()
