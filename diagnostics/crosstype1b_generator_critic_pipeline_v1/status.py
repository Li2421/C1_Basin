"""Unified read-only progress inspection; optional local status receipt."""
from core import *
from resource_guard import queue


def main():
    phases={}
    for phase in ('pilot','remaining','new_evaluation'):
        missing=HERE/f'{phase}_MISSING_CASES.json'
        phases[phase]=dict(raw_results=sum(1 for _ in (HERE/phase/'raw').glob('*.json')),
            preflight_missing=len(read(missing)) if missing.exists() else None,
            postflight_complete=(HERE/f'{phase}_COMPLETE.receipt.json').exists())
    statuses={}
    for key in ('LIVE_STATUS','CRITIC_LIVE_STATUS','CONFIRMATION_LIVE_STATUS'):
        if (HERE/f'{key}.json').exists():statuses[key]=read(HERE/f'{key}.json')
    result=dict(utc=utc(),phases=phases,statuses=statuses,
        critic_fits_complete=sum(1 for _ in (HERE/'critic_fits').glob('*/*/COMPLETE.receipt.json')),
        frozen=(HERE/'FROZEN_PIPELINE.receipt.json').exists(),
        new_state_choices_sealed=(HERE/'NEW_EVAL_CHOICES.receipt.json').exists(),
        delivered=(HERE/'FINAL_DELIVERY.receipt.json').exists(),
        errors={f.name:read(f) for f in HERE.glob('*ERROR.json')},
        resources={k:v for k,v in queue().items() if k not in ('rows','raw')})
    print(json.dumps(clean(result),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
