import json,run
if __name__=='__main__':
 sp=json.load(open('SPLIT.json'))
 S=[s for f in ['scenes_design.json','scenes_extra.json'] for s in json.load(open(f))]
 S={s['name']:s for s in S};keep=[S[n] for n in sp['train']+sp['val']]
 json.dump(keep,open('scenes_set.json','w'))
 tasks=[(n,ci,si,ai) for n in sp['val'] for ci in range(3) for si in range(8,24) for ai in range(49)]+[(n,ci,si,ai) for n in sp['train'] for ci in range(3) for si in range(8,16) for ai in range(49)]
 tasks.sort(key=lambda t:0 if t[0].startswith('hs_obst') else 1)
 run.run(tasks,'scenes_set.json','fresh.jsonl',16)
