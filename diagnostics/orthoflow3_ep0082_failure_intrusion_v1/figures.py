#!/usr/bin/env python3
from core import *
from html import escape
def circles(parts,xy,labels,x0,y0,w=330,h=240,polyline=None):
 lo=xy.min(0)-.015;hi=xy.max(0)+.015;pts=(xy-lo)/np.maximum(hi-lo,1e-10)*[w,h];parts.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#999"/>')
 if polyline is not None:
  qq=(polyline-lo)/np.maximum(hi-lo,1e-10)*[w,h];parts.append('<polyline points="'+' '.join(f'{x0+x:.2f},{y0+h-y:.2f}' for x,y in qq)+'" fill="none" stroke="#666" stroke-dasharray="4 3"/>')
 for (x,y),label in zip(pts,labels):
  x+=x0;y=y0+h-y
  if label=='S':parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#2166ac" opacity=".75"/>')
  elif label=='F':parts.append(f'<path d="M{x-3:.2f},{y-3:.2f}l6,6m0,-6l-6,6" stroke="#b2182b" stroke-width="1.5"/>')
  else:parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="white" stroke="#999"/>')
 parts.append(f'<text x="{x0}" y="{y0+h+18}" font-size="10">x[{lo[0]:.3f},{hi[0]:.3f}],y[{lo[1]:.3f},{hi[1]:.3f}]</text>')
def start(title,height=680):return [f'<svg xmlns="http://www.w3.org/2000/svg" width="850" height="{height}" viewBox="0 0 850 {height}">','<rect width="100%" height="100%" fill="white"/>','<g font-family="sans-serif">',f'<text x="25" y="25" font-size="17">{escape(title)}</text>','<text x="25" y="48" font-size="12">Blue: observed B63; red: observed non-B63; hollow/blank: UNKNOWN.</text>']
def save(name,p):p+=['</g></svg>'];(H/name).write_text('\n'.join(p))
def main():
 rows=read(H/'exact_q64_ep0082.csv');lookup={r['eta_key']:r for r in rows};Z=np.array([norm(r) for r in rows]);L=np.array(['S' if r['B63']=='True' else 'F' for r in rows]);c,R=frame();Y=(Z-c)@R;regions=json.load(open(H/'region_definitions.json'))['regions']
 p=start('ep0082: observed normalized eta (3-D isometric and native slices)');iso=np.array([[.87,-.3],[-.87,-.3],[0,1.]])
 for j,(xy,title) in enumerate([(Z@iso,'Isometric, not occupancy reconstruction'),(Z[:,[0,1]],'native eta1 / eta2'),(Z[:,[0,2]],'native eta1 / eta3'),(Z[:,[1,2]],'native eta2 / eta3')]):
  x=35+(j%2)*420;y=90+(j//2)*290;p.append(f'<text x="{x}" y="{y-12}" font-size="12">{title}</text>');circles(p,xy,L,x,y)
 save('eta_3d_success_failure.svg',p)
 for name,axes,fixed in [('tangential_slices',[0,1],2),('mixed_s1_n_slices',[0,2],1),('mixed_s2_n_slices',[1,2],0)]:
  p=start(name+' — approximate +/-0.05 slice bands, not exact planes')
  for j,reg in enumerate(regions):
   q=((np.array(reg['eta'])-AFF)/SCALE-c)@R;mask=abs(Y[:,fixed]-q[fixed])<=.05;x=35+j%2*420;y=90+j//2*290;p.append(f'<text x="{x}" y="{y-12}" font-size="12">region{reg["region_id"]}; fixed coordinate={q[fixed]:.3f}</text>');circles(p,Y[mask][:,axes],L[mask],x,y)
  save(name+'.svg',p)
 corridor_rows=read(H/'candidate_failure_corridors.csv');corridor_ids=list(dict.fromkeys(r['corridor_id'] for r in corridor_rows))
 p=start('Candidate failure corridors — dashed gaps are UNKNOWN',height=95+290*((len(corridor_ids)+1)//2))
 for j,cid in enumerate(corridor_ids):
  rr=[r for r in corridor_rows if r['corridor_id']==cid];zz=np.array([(np.array(json.loads(r['eta']))-AFF)/SCALE for r in rr]);labels=[('S' if lookup[r['eta_key']]['B63']=='True' else 'F') if r['eta_key'] in lookup else '?' for r in rr]
  x=35+j%2*420;y=90+j//2*290;p.append(f'<text x="{x}" y="{y-12}" font-size="12">{cid}; isometric path</text>');circles(p,zz@iso,labels,x,y,polyline=zz@iso)
 save('failure_corridor_view.svg',p)
 if (H/'conditional_slice_results.csv').exists():
  lines=read(H/'conditional_slice_results.csv');stats=read(H/'transition_statistics.csv');p=start('Exact conditional line profiles — gaps UNKNOWN',height=1250)
  for j,s in enumerate(stats):
   rr=sorted({r['eta_key']:r for r in lines if r['line_id']==s['line_id']}.values(),key=lambda r:float(r['t']));xy=np.array([[float(r['t']),float(r['Q64'])] for r in rr]);labels=['S' if r['B63']=='True' else 'F' for r in rr];x=35+j%2*420;y=85+j//2*185;p.append(f'<text x="{x}" y="{y-10}" font-size="11">{s["line_id"]}; transitions={s["transition_count"]}</text>');circles(p,xy,labels,x,y,h=135)
  save('transition_line_examples.svg',p)
 print('Saved observed-only SVG views; unknown space not labeled successful.')
if __name__=='__main__':main()
