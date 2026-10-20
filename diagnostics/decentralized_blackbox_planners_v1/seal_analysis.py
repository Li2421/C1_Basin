from common import *
G=guard()
seal('ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json',[HERE/'analyze.py',HERE/'metrics.py',HERE/'stats.py',HERE/'mechanism_core.py'],before_opening_main_labels=True,before_pilot_labels=True,guard=guard())
