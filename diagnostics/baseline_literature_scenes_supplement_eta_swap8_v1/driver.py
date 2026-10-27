"""Deployment entry for the user-corrected coordinator; no module patching."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent/"driver_user_fix_20261023.py"),run_name="__main__")
