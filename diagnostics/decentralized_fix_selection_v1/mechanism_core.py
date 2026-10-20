"""Local analysis support. No simulator, controller, training, or DB import."""
import os,sys,pathlib,json,hashlib,datetime
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];D=ROOT/'diagnostics'
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
for k,v in [('TMPDIR','tmp'),('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache')]:
 p=HERE/v;p.mkdir(exist_ok=True);os.environ[k]=str(p)
sys.path.insert(0,str(ROOT))
import numpy as np
import jax
jax.config.update('jax_enable_x64',False)
def precision():
 jax.config.update('jax_enable_x64',False);assert not jax.config.x64_enabled;return dict(jax_enable_x64=False,numpy_geometry='float64',backend='CPU only; no model queries')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def write(p,obj):
 p=pathlib.Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def seal(name,files,**kw):
 assert not (HERE/name).exists();write(HERE/name,dict(utc=utc(),files={str(pathlib.Path(p).resolve()):sha(p) for p in files},**kw))
def prereg():
 for p,h in read(HERE/'PREREGISTRATION.receipt.json')['files'].items():assert sha(p)==h,p
FULL={'expert':'集中式专家（特权参照；expert）','TC':'联合教师加集中式安全层（TC）','TD':'联合教师加分布式安全层（TD）','V0':'原学生（离线蒸馏、各自噪声；V0）','V1':'闭环重标注学生（各自噪声；V1）','V3':'闭环重标注加共享随机种子学生（V3）','Z':'零动作控制器加分布式安全层（Z）'}
ARMS=['nominal only','safety baseline','F10','F13','S300','S500','RHR-fair']
ARM_FULL={'nominal only':'只跑学习型控制器、不加安全层与修正（nominal only）','safety baseline':'只加安全层、修正系数为零（safety baseline）','F10':'全程使用第10组修正参数（F10）','F13':'全程使用第13组修正参数（F13）','S300':'前300步使用第13组、之后第10组修正参数的时间表（S300）','S500':'前500步使用第13组、之后第10组修正参数的时间表（S500）','RHR-fair':'公平调参右手规则：无进展触发、2秒、0.25米每秒（RHR-fair）','expert':'集中式专家原生规划（expert）'}
