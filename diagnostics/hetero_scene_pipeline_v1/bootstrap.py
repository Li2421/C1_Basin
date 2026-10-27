"""First import: private static graph and output/cache environment only."""
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,str(HERE/'backend'))
# Root experiment.py always wins over historical project locations.
sys.path.insert(0,str(HERE))
import experiment
