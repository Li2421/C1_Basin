from core import *
from guard import guard
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
from poc.field_controller import Runtime
from ring_exchange.environment import RingExchangeEnv
from ring_exchange.local_frame import local_observation
import jax
LOCAL=[8,10,12,13]
guard()
def teacher(index):
 r=read(HERE/'TEACHERS.json')[index];assert sha(r['checkpoint'])==r['checkpoint_sha256']
 v=read(D/'decentralized_pipeline_v1/main_protocol.json')['base_profiles'][r['name']]['payload']
 # make_runtime('ring_exchange') constructs FourRingRuntime, whose only override
 # concerns optional trace rendering. All native_fn/trace_fn/config are Runtime's.
 rt=Runtime('ring_exchange') if index==8 else RingFlowAdapter(v['flow_checkpoint_path'],read(v['adapter_manifest_path']),v['observation_frame'])
 assert sha(rt.checkpoint)==r['checkpoint_sha256'];guard();return rt
def dataset(index,split):
 r=read(HERE/'TEACHERS.json')[index];p=D/'teacher_student_macflow_v1/data'/r['name']
 assert (p/'READY.json').exists()
 return np.load(p/f'{split}_local.npy'),np.load(p/f'{split}_teacher.npy'),np.load(p/'normalization.npz')
