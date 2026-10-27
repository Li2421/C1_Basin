from audit_supplement import *
import subprocess
repo=HERE/'external/hjbgnn';commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
paths=[repo/'hjbgnn/env/crazyflie.py',repo/'hjbgnn/env/obstacle.py',repo/'settings.yaml',repo/'README.md',HERE/'papers/hjbgnn_v2.pdf',HERE/'papers/hjbgnn_v2.txt',HERE/'papers/page8.png']
write(HERE/'EXTERNAL_SOURCES.json',dict(utc=datetime.now(timezone.utc).isoformat(),paper_url='https://arxiv.org/pdf/2506.22117v2',paper_version='2506.22117v2 13May2026',project_url='https://nus-core.github.io/assets/standalone/HJB-GNN/index.html',repository_url='https://github.com/hublan24/HJB-GNN',commit=commit,files={str(p):sha(p) for p in paths},external_code_used='Read-only parameters only; no execution/import/install',figure_reading=dict(figure=7,pdf_page=8,dpi=420,calibrated_metric_layout=False,reason='Uncalibrated real photograph with no metric axes; declared prompt fallback used')))
