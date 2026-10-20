"""Resume postprocessing only after the renderer environment failure."""
from support import *
import subprocess
G=guard()
def call(python,script,*args):
 r=subprocess.run([python,'-B',str(HERE/script),*args],capture_output=True,text=True)
 print(r.stdout+r.stderr,flush=True)
 assert r.returncode==0,(script,r.returncode)
def main():
 prereg()
 for p,h in read(HERE/'ANALYSIS_CODE_FROZEN.receipt.json')['files'].items():assert sha(p)==h
 seal('RENDER_PARENT.receipt.json',[HERE/'PLOTTING_AMENDMENT.json',HERE/'PLOTTING_AMENDMENT.patch',HERE/'plots_render_cpu.py',HERE/'results.json'],guard=guard(),no_JAX_in_child=True)
 call('/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python','plots_render_cpu.py')
 call(sys.executable,'extra_dependency.py','END')
 call(sys.executable,'integrity.py','END')
 call(sys.executable,'report.py')
 call(sys.executable,'verify.py')
if __name__=='__main__':main()
