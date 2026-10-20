"""Documentation-only evidence-grade qualification; numerical estimates unchanged."""
from delivery import main as original_delivery
from common import *
import csv

def main():
 original_delivery();results=read(HERE/'results.json');grades={}
 for label,r in results.items():
  if not isinstance(r,dict) or 'methods' not in r:continue
  g=list(r['evidence_grade_counts']);grades[label]='CONFIRMATORY prospective' if all(x.startswith('CONFIRMATORY') for x in g) else 'DEVELOPMENT historical reanalysis' if all(x.startswith('DEVELOPMENT') for x in g) else 'MIXED historical DEVELOPMENT / prospective CONFIRMATORY'
 path=HERE/'AXIS_TABLE.csv';rows=list(csv.DictReader(path.open()));cols=list(rows[0])
 for row in rows:row['evidence']=grades.get(row['subaxis'],'GAP')
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
 md=HERE/'AXIS_TABLE.md';old=md.read_text();footer=old[old.index('\n\n'):];md.write_text('| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(row[c] for c in cols)+' |' for row in rows)+footer)
 mm=read(HERE/'METHOD_METRICS.json')
 for row in mm:row['evidence']=grades[row['subaxis']]
 write(HERE/'METHOD_METRICS.json',mm);path=HERE/'METHOD_METRICS.csv';rows=list(csv.DictReader(path.open()));cols=list(rows[0])
 for row in rows:row['evidence']=grades[row['subaxis']]
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
