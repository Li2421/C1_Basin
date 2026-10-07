from __future__ import annotations
import argparse,csv,hashlib,json,os,re,sqlite3
from pathlib import Path
from typing import Any
from .rollout_db import ROOT,connect,initialize,canonical,eta_identity,uid

REPO=ROOT.parent
DIAG=REPO/'diagnostics'
BASIS_SHA='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38'
TOY_FLOW_SHA='8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32'
DB_FLOW_SHA='6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd'

def file_sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def jhash(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def truth(x): return str(x).lower() in ('1','true','yes','success') if not isinstance(x,bool) else x
def num(x,default=None):
    try:return float(x)
    except:return default
def intval(x,default=None):
    try:return int(float(x))
    except:return default

class Ingestor:
  def __init__(self,root=REPO):
    initialize();self.root=Path(root);self.con=connect();self.stats={k:0 for k in ['files_scanned','experiment_dirs','seed_imported','aggregate_imported','duplicates','conflicts','ambiguous','summary_only','rows_seen']};self.experiments=set()
  def experiment(self,p,override=None):
    p=Path(p).resolve()
    if override:
      eu=str(override);found=self.con.execute('SELECT name,path FROM experiment WHERE experiment_uid=?',(eu,)).fetchone()
      if found:return eu,found['name'],found['path']
      name=eu;ep=str(p.parent)
      self.con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,metadata_json) VALUES(?,?,?,?)',(eu,name,ep,canonical({'journal_placeholder':True})))
      return eu,name,ep
    else:
      try: rel=p.relative_to(DIAG);name=rel.parts[0];ep=str(DIAG/name)
      except ValueError:
        name=p.parent.name;ep=str(p.parent)
    eu=uid('exp',{'path':ep});self.experiments.add(ep)
    proto=None
    for q in (Path(ep)/'protocol.md',Path(ep)/'candidate_protocol.md',Path(ep)/'final_report.md'):
      if q.exists():proto=file_sha(q);break
    self.con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,metadata_json) VALUES(?,?,?,?,?)',(eu,name,ep,proto,canonical({'historical_ingest':True})))
    return eu,name,ep
  def source(self,p,classification,rows_seen=0,experiment_override=None):
    eu,name,ep=self.experiment(p,experiment_override);sp=str(Path(p).resolve());su=uid('src',{'path':sp});sh=file_sha(p)
    old=self.con.execute('SELECT sha256 FROM source_file WHERE path=?',(sp,)).fetchone()
    if old and old['sha256']!=sh:
      cu=uid('conflict',{'path':sp,'old':old['sha256'],'new':sh});self.con.execute('INSERT OR IGNORE INTO conflict VALUES(?,?,?,?,?,?,?,CURRENT_TIMESTAMP)',(cu,'SOURCE_MUTATION',sp,canonical(dict(old)),canonical({'sha256':sh}),sp,'CONFLICT_QUARANTINED'));self.stats['conflicts']+=1
    self.con.execute('''INSERT OR IGNORE INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen)
      VALUES(?,?,?,?,?,?,?)''',(su,eu,sp,sh,Path(p).suffix.lower(),classification,rows_seen))
    return su,eu,name
  def scenario(self,r,p):
    s=str(r.get('scenario') or '')
    text=(str(p)+' '+str(r.get('state_id',''))+' '+str(r.get('episode_id',''))+' '+s).lower()
    if 'doublebottleneck' in text or 'double_bottleneck' in text or str(r.get('state_id','')).startswith('DB_'):name='DoubleBottleneck_4A';code='db4a_frozen_v1'
    elif any(x in text for x in ['orthoflow','giveway','give_way','toy']):name='ToyGiveWay';code='toy_giveway_frozen_v1'
    elif s:name=s;code='declared_'+jhash(s)[:16]
    else:name='UNKNOWN';code='unknown'
    suid=uid('scn',{'name':name,'code':code});self.con.execute('INSERT OR IGNORE INTO scenario VALUES(?,?,?,?,CURRENT_TIMESTAMP)',(suid,name,code,canonical({'inferred_from':str(p)})));return suid,name
  def state(self,r,p,scenario_uid,experiment_uid):
    alias=str(r.get('state_id') or r.get('episode_id') or r.get('case_id') or r.get('family_id') or (f"rngns:{r['rng_namespace']}" if r.get('rng_namespace') is not None else 'UNKNOWN'))
    sg=str(r.get('source_group') or r.get('family_id') or '')
    physical={k:r[k] for k in ['initial_positions','initial_velocities','goals','obstacles','geometry','route_configuration'] if k in r}
    h=r.get('h0') or r.get('h_feature') or None;cond=r.get('h_conditioning_identifier') or r.get('h_sha256')
    if physical: content=jhash(physical);quality='CONTENT_EXACT'
    elif cond:content=str(cond);quality='CONDITIONING_EXACT'
    elif sg:content=jhash({'scenario':scenario_uid,'source_group':sg});quality='SOURCE_GROUP_STABLE'
    elif r.get('rng_namespace') is not None:content=jhash({'scenario':scenario_uid,'rng_namespace':r['rng_namespace'],'source_file_parent':str(Path(p).parent.parent)});quality='SOURCE_GROUP_STABLE'
    else:content=jhash({'scenario':scenario_uid,'alias':alias,'experiment':experiment_uid});quality='ALIAS_ONLY'
    sid=uid('state',{'scenario':scenario_uid,'content':content})
    self.con.execute('''INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,h0_json,goals_geometry_json,provenance_json,identity_quality)
      VALUES(?,?,?,?,?,?,?,?,?)''',(sid,scenario_uid,sg or None,content,canonical(physical) if physical else None,canonical(h) if h is not None else None,
      canonical({k:physical[k] for k in physical if k in ('goals','obstacles','geometry','route_configuration')}) if physical else None,canonical({'alias':alias,'path':str(p)}),quality))
    self.con.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(scenario_uid,alias,sid,experiment_uid));return sid,quality
  def eta(self,r):
    e=r.get('eta',r.get('theta'))
    if isinstance(e,str):
      try:e=json.loads(e)
      except:
        try:e=[float(x) for x in re.split('[,; ]+',e.strip('[]() ')) if x]
        except:return None
    if e is None and all(k in r for k in ('eta1','eta2','eta3')):e=[r['eta1'],r['eta2'],r['eta3']]
    if not isinstance(e,(list,tuple)) or len(e)!=3:return None
    try:eid,v,hx=eta_identity(e)
    except:return None
    norm=[(v[0]-.875)/.75,v[1],(v[2]-.375)/.75]
    self.con.execute('INSERT OR IGNORE INTO eta VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP)',(eid,*v,canonical(norm),hx));return eid,v
  def controller(self,r,p,scenario_uid,scenario_name,state_quality):
    rep_declared=str(r.get('representation') or r.get('controller') or 'unknown')
    is_ortho=('orthoflow3' in str(p).lower() or 'P1-OrthoFlow3' in rep_declared or ('eta' in r and scenario_name in ('ToyGiveWay','DoubleBottleneck_4A')))
    flow=DB_FLOW_SHA if scenario_name=='DoubleBottleneck_4A' else (TOY_FLOW_SHA if scenario_name=='ToyGiveWay' else None)
    # Modern OrthoFlow directories share the frozen controller chain. Unknown/legacy
    # representations remain searchable but are never automatic EXACT_REUSE.
    # A historical/diagnostic directory can contain a pure Flow comparator.
    # Its control chain must never share the OrthoFlow3 cache fingerprint.
    modern=is_ortho and str(r.get('controller_kind','')).upper()!='MAC_ONLY' and ('orthoflow3_' in str(p).lower() or rep_declared=='P1-OrthoFlow3')
    # Experiment-local labels such as mode_00/fixed/selector do not change the
    # frozen physical controller once their exact eta is part of the cache key.
    rep='P1-OrthoFlow3' if modern else rep_declared
    safety='certified_hard_projection_v1' if modern else None
    cfg={'scenario':scenario_name,'flow_sha256':flow if modern else r.get('flow_checkpoint_sha256'),'basis_sha256':BASIS_SHA if modern else r.get('orthoflow3_sha256'),
         'safety':safety,'representation':rep,'horizon':r.get('horizon') or r.get('max_steps'),'dt':r.get('dt'),
         'success_semantics':'frozen_success_deadlock_timeout_v1' if modern else r.get('success_semantics_version'),
         'conditioning':'true_t0_latched_eta_v1' if modern else r.get('conditioning_version'),
         'rng':(r.get('rng_semantics_version') or ({'name':'matched_future_index_v1','future_root':r.get('future_root_seed') or r.get('future_root') or (2026092907 if scenario_name=='DoubleBottleneck_4A' else 2026092811)} if 'future_index' in r else ({'name':'seed_rollout_foldin_v1'} if 'seed' in r and 'rollout_id' in r else None)))}
    explicit_mac=(str(r.get('controller_kind','')).upper()=='MAC_ONLY' and
                  all(cfg[k] is not None for k in ('flow_sha256','horizon','dt','success_semantics','conditioning','rng')))
    quality='EXACT_PROFILE' if (modern or explicit_mac) and state_quality!='ALIAS_ONLY' and cfg['rng'] else ('PARTIAL_PROFILE' if modern else 'AMBIGUOUS')
    cid=uid('ctl',cfg)
    self.con.execute('''INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',(cid,scenario_uid,cfg['flow_sha256'],cfg['basis_sha256'],cfg['safety'],str(cfg['horizon']) if cfg['horizon'] is not None else None,str(cfg['dt']) if cfg['dt'] is not None else None,cfg['success_semantics'],cfg['conditioning'],canonical(cfg['rng']) if cfg['rng'] is not None else None,canonical(cfg),quality));return cid,quality
  def seed(self,r):
    if 'future_index' in r:
      # Root/namespace belong to controller/state fingerprints. Keeping the
      # logical index alone lets old rows lacking redundant root metadata match
      # future journals without weakening semantic compatibility.
      x={'future_index':intval(r['future_index'])}
      return canonical(x),x
    if 'continuation_seed' in r:x={'continuation_seed':r['continuation_seed']};return canonical(x),x
    if 'seed' in r and 'rollout_id' in r:x={'seed':r['seed'],'rollout_id':r['rollout_id']};return canonical(x),x
    if 'seed' in r:x={'seed':r['seed']};return canonical(x),x
    return None,None
  def outcomes(self,r):
    out=str(r.get('outcome') or r.get('termination') or '').lower();success=truth(r.get('success',out=='success'))
    dead=truth(r.get('deadlock',out in ('deadlock','safe_deadlock')));tout=truth(r.get('timeout',out=='timeout'))
    coll=truth(r.get('collision',False)) or truth(r.get('wall_collision',False)) or truth(r.get('agent_collision',False)) or out=='collision'
    numfail=truth(r.get('numerical_failure',False)) or r.get('scientific_outcome_valid') is False or bool(r.get('execution_error')) or 'numerical' in out
    return int(success),int(dead),int(tout),int(coll),int(numfail),out
  def insert_seed(self,r,p,su,eu,line):
    eta=self.eta(r);seed_key,seed_obj=self.seed(r)
    if not eta or seed_key is None:return 'ambiguous'
    scid,scname=self.scenario(r,p);sid,sq=self.state(r,p,scid,eu);cid,cq=self.controller(r,p,scid,scname,sq);eid,_=eta
    vals=self.outcomes(r);identity={'state':sid,'eta':eid,'controller':cid,'seed':seed_key};rid=uid('roll',identity);rawh=jhash(r)
    existing=self.con.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?',(sid,eid,cid,seed_key)).fetchone()
    if existing:
      core=(existing['success'],existing['deadlock'],existing['timeout'],existing['collision'],existing['numerical_failure'])
      if core!=vals[:5]:
        cu=uid('conflict',{'key':identity,'existing':core,'incoming':vals[:5]});self.con.execute('INSERT OR IGNORE INTO conflict VALUES(?,?,?,?,?,?,?,CURRENT_TIMESTAMP)',(cu,'ROLLOUT',canonical(identity),canonical(dict(existing)),canonical(r),str(p),'CONFLICT_QUARANTINED'));self.con.execute('UPDATE rollout SET conflict_quarantined=1 WHERE rollout_uid=?',(existing['rollout_uid'],));self.stats['conflicts']+=1;return 'conflict'
      self.con.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',(existing['rollout_uid'],su,line));self.stats['duplicates']+=1;return 'duplicate'
    compat='EXACT_REUSE' if sq in ('CONTENT_EXACT','CONDITIONING_EXACT','SOURCE_GROUP_STABLE') and cq=='EXACT_PROFILE' else 'AMBIGUOUS'
    self.con.execute('''INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,j_def,min_wall_distance,min_agent_distance,outcome,experiment_uid,original_source_file,timestamp,compatibility_quality,raw_record_hash)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(rid,sid,eid,cid,seed_key,canonical(seed_obj),*vals[:5],num(r.get('episode_length') or r.get('episode_steps') or r.get('continuation_steps')),num(r.get('J_def')),num(r.get('minimum_wall_clearance')),num(r.get('minimum_agent_clearance')),vals[5],eu,str(p),r.get('timestamp'),compat,rawh))
    self.con.execute('INSERT INTO rollout_source VALUES(?,?,?)',(rid,su,line));self.stats['seed_imported']+=1
    if compat!='EXACT_REUSE':self.stats['ambiguous']+=1
    return 'imported'
  def aggregate_counts(self,r,p):
    n=intval(r.get('trials') or r.get('n_trials') or r.get('N') or r.get('num_trials'))
    s=intval(r.get('successes') or r.get('n_success') or r.get('success_count'))
    if n is None:
      for key,N in [('Q64',64),('q64',64),('Q32',32),('q32',32),('Q16',16),('q16',16)]:
        if key in r:n=N;q=num(r[key]);s=int(round(q*N)) if q is not None and q<=1 else intval(q);break
    if n is None or s is None or n<=0 or not (0<=s<=n):return None
    return n,s
  def insert_aggregate(self,r,p,su,eu,line):
    eta=self.eta(r);counts=self.aggregate_counts(r,p)
    if not eta or not counts:return 'ambiguous'
    scid,scname=self.scenario(r,p);sid,sq=self.state(r,p,scid,eu);cid,cq=self.controller(r,p,scid,scname,sq);eid,_=eta;n,s=counts
    ev='Q64' if n==64 else ('Q32' if n==32 else ('Q16' if n==16 else f'Q{n}'))
    cert=int(n>=16 and n-s<=1);prov={'path':str(p),'line':line,'row':r};identity={'state':sid,'eta':eid,'controller':cid,'n':n,'s':s,'dead':intval(r.get('deadlock')),'timeout':intval(r.get('timeout')),'collision':intval(r.get('collision'))}
    aid=uid('agg',identity)
    before=self.con.total_changes
    self.con.execute('''INSERT OR IGNORE INTO aggregate_evidence VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)''',(aid,sid,eid,cid,n,s,identity['dead'],identity['timeout'],identity['collision'],0,ev,cert,canonical(prov),eu,str(p),0))
    self.con.execute('INSERT OR IGNORE INTO aggregate_source VALUES(?,?,?)',(aid,su,line))
    if self.con.total_changes>before:self.stats['aggregate_imported']+=1;return 'imported'
    self.stats['duplicates']+=1;return 'duplicate'
  def ingest_jsonl(self,p,experiment_override=None,journal=False):
    p=Path(p);su,eu,_=self.source(p,'SEED_EXACT_CANDIDATE',experiment_override=experiment_override);seen=imp=dup=amb=0
    with p.open(errors='replace') as f:
      for line_no,line in enumerate(f,1):
       if not line.strip():continue
       seen+=1;self.stats['rows_seen']+=1
       try:r=json.loads(line)
       except:amb+=1;continue
       if not isinstance(r,dict):amb+=1;continue
       if not any(k in r for k in ('success','outcome','termination')):amb+=1;continue
       st=self.insert_seed(r,p,su,eu,line_no)
       if st=='imported':imp+=1
       elif st=='duplicate':dup+=1
       else:amb+=1
       if seen%10000==0:self.con.commit()
    prior=self.con.execute('SELECT classification FROM source_file WHERE source_uid=?',(su,)).fetchone()
    cls='SEED_EXACT' if (imp or dup or (prior and prior['classification']=='SEED_EXACT')) else 'SUMMARY_ONLY'
    self.con.execute('UPDATE source_file SET classification=?,rows_seen=?,rows_imported=?,rows_duplicate=?,rows_ambiguous=? WHERE source_uid=?',(cls,seen,imp,dup,amb,su));self.con.commit();return imp
  def ingest_csv(self,p):
    p=Path(p)
    try:
      with p.open(newline='',errors='replace') as f:header=next(csv.reader(f),[])
    except:return 0
    hs=set(header);has_eta=('eta' in hs or {'eta1','eta2','eta3'}<=hs or 'theta' in hs);has_state=bool(hs&{'state_id','episode_id','case_id','family_id'});has_counts=bool(hs&{'trials','n_trials','Q64','q64','Q32','q32','Q16','q16'})
    if not (has_eta and has_state and has_counts):return 0
    su,eu,_=self.source(p,'AGGREGATE_EXACT_CANDIDATE');seen=imp=dup=amb=0
    try:
      with p.open(newline='',errors='replace') as f:
       for line_no,r in enumerate(csv.DictReader(f),2):
        seen+=1;self.stats['rows_seen']+=1;st=self.insert_aggregate(r,p,su,eu,line_no)
        if st=='imported':imp+=1
        elif st=='duplicate':dup+=1
        else:amb+=1
        if seen%10000==0:self.con.commit()
    except (csv.Error,UnicodeError):amb+=1
    prior=self.con.execute('SELECT classification FROM source_file WHERE source_uid=?',(su,)).fetchone()
    cls='AGGREGATE_EXACT' if (imp or dup or (prior and prior['classification']=='AGGREGATE_EXACT')) else 'SUMMARY_ONLY'
    self.con.execute('UPDATE source_file SET classification=?,rows_seen=?,rows_imported=?,rows_duplicate=?,rows_ambiguous=? WHERE source_uid=?',(cls,seen,imp,dup,amb,su));self.con.commit();return imp
  def register_summary(self,p):
    self.source(p,'SUMMARY_ONLY');self.stats['summary_only']+=1;self.con.commit()
  def scan(self):
    dirs=[p for p in DIAG.iterdir() if p.is_dir()];self.stats['experiment_dirs']=len(dirs)
    excluded_markers=('invalid_feature','invalid_cache','feature_mismatch','quarantine_batchshape','quarantine_debug','quarantined_pretransition','cancelled','canceled','excluded_rollout','quarantined_bad')
    for i,p in enumerate(sorted(self.root.rglob('*.jsonl'))):
      if ROOT in p.parents:continue
      self.stats['files_scanned']+=1
      if any(x in str(p).lower() for x in excluded_markers):self.register_summary(p);continue
      # Inspect first nonempty record before paying full parsing cost.
      try:
       first=next((json.loads(x) for x in p.open(errors='replace') if x.strip()),None)
      except: first=None
      if isinstance(first,dict) and any(k in first for k in ('eta','theta','eta1')) and any(k in first for k in ('success','outcome','termination')):self.ingest_jsonl(p)
      else:self.register_summary(p)
    for p in sorted(self.root.rglob('*.csv')):
      if ROOT in p.parents:continue
      self.stats['files_scanned']+=1
      if not self.ingest_csv(p):
       # Register only likely scientific result CSVs, not every training history.
       if any(x in p.name.lower() for x in ('q64','q32','q16','rollout','result','evidence','coverage')):self.register_summary(p)
    for p in sorted(self.root.rglob('*.json')):
      if ROOT in p.parents:continue
      if any(x in p.name.lower() for x in ('manifest','report','decision','summary')):self.register_summary(p)
    self.finish();return self.stats
  def finish(self):self.stats['experiment_dirs']=len(self.experiments) or self.stats.get('experiment_dirs',0);self.con.commit()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(REPO));a=ap.parse_args();ing=Ingestor(a.root);stats=ing.scan();(ROOT/'audits').mkdir(exist_ok=True);(ROOT/'audits'/'ingest_stats.json').write_text(json.dumps(stats,indent=2,sort_keys=True)+'\n');print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
