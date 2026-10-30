"""Copy old paper labels and their referenced traces without importing old code."""
import guard
from guard import HERE,ROOT
import prepare_static as snapshot
import datetime,json

def main():
    base=ROOT/'diagnostics/scene_types_oracle_coverage_v3'
    files=sorted((base/'paper_after_four_column/labels').glob('*.json'))
    uids=set();label_files=[]
    def gather(value):
        if isinstance(value,dict):
            uid=value.get('rollout_uid')
            if isinstance(uid,str)and uid.startswith('roll_'):uids.add(uid)
            for child in value.values():gather(child)
        elif isinstance(value,list):
            for child in value:gather(child)
    for file in files:
        relative=str(file.relative_to(ROOT));snapshot.copy(relative)
        target=HERE/'inputs'/relative;gather(json.loads(target.read_text()));label_files.append(str(target.relative_to(HERE)))
    records=[];missing=[]
    for uid in sorted(uids):
        raw=base/'raw'/(uid+'.json')
        if not raw.exists():missing.append(dict(rollout_uid=uid,missing='raw'));continue
        relative=str(raw.relative_to(ROOT));snapshot.copy(relative)
        rawcopy=HERE/'inputs'/relative;value=json.loads(rawcopy.read_text());trace=value.get('trace');tracecopy=None
        if trace:
            source=__import__('pathlib').Path(trace)
            if source.exists():
                relative=str(source.relative_to(ROOT));snapshot.copy(relative);tracecopy=str((HERE/'inputs'/relative).relative_to(HERE))
            else:missing.append(dict(rollout_uid=uid,missing='trace',path=trace))
        records.append(dict(rollout_uid=uid,raw=str(rawcopy.relative_to(HERE)),trace=tracecopy,original_trace=trace))
    meta=json.loads(snapshot.MANIFEST.read_text());meta['files']=snapshot.records;meta['last_history_copy_at']=datetime.datetime.now().astimezone().isoformat();snapshot.MANIFEST.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    output=dict(label_files=label_files,requested_unique_rollouts=len(uids),copied_rollouts=records,missing=missing,new_rollouts=0)
    (HERE/'PAPER_HISTORY_COPY.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print('Copied historical paper labels:',len(files),'raw episodes:',len(records),'traces:',sum(r['trace']is not None for r in records),'missing:',len(missing),'new rollouts0')

if __name__=='__main__':main()
