#!/usr/bin/env python3
"""Dependency-free scientific SVG scatter panels. Blank space is UNKNOWN."""
from audit import *
from html import escape
def main():
 rows=[r for r in read(HERE/'exact_q64_inventory.csv') if inside((eta(r)-AFF)/SCALE)[0]];dest=HERE/'figures';dest.mkdir(exist_ok=True)
 paths=[]
 for sid in sorted({r['state_id'] for r in rows}):
  rr=[r for r in rows if r['state_id']==sid]
  if len(rr)<30:continue
  Z=np.array([(eta(r)-AFF)/SCALE for r in rr]);good=np.array([r['B63']=='True' for r in rr]);_,_,V=np.linalg.svd(Z[good]-Z[good].mean(axis=0),full_matrices=False)
  local=(Z-Z[good].mean(axis=0))@V.T
  panels=[(Z[:,[0,1]],'native normalized eta1 / eta2'),(Z[:,[0,2]],'native normalized eta1 / eta3'),(Z[:,[1,2]],'native normalized eta2 / eta3'),
    (local[:,:2],'PCA tangent1 / tangent2'),(local[:,[0,2]],'PCA tangent1 / normal'),(Z@np.array([[.87,-.3],[-.87,-.3],[0,1.]]),'3-D isometric projection (no occlusion claim)')]
  svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1170" height="800" viewBox="0 0 1170 800">','<rect width="100%" height="100%" fill="white"/>',
   f'<text x="25" y="26" font-family="sans-serif" font-size="17">{escape(sid)} — observed exact Q64 only</text>',
   '<text x="25" y="50" font-family="sans-serif" font-size="13">Blue circle: B63. Red cross: non-B63. Blank space: UNOBSERVED / UNKNOWN.</text>']
  for j,(xy,title) in enumerate(panels):
   x0=45+(j%3)*385;y0=95+(j//3)*345;lo=xy.min(axis=0)-.035;hi=xy.max(axis=0)+.035;size=np.array([320,260]);q=(xy-lo)/(hi-lo)*size
   svg += [f'<text x="{x0}" y="{y0-15}" font-family="sans-serif" font-size="12">{escape(title)}</text>',f'<rect x="{x0}" y="{y0}" width="320" height="260" fill="none" stroke="#888"/>']
   for i,(x,y) in enumerate(q):
    x+=x0;y=y0+260-y
    if good[i]:svg.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2.8" fill="#2166ac" opacity=".72"/>')
    else:svg.append(f'<path d="M{x-3:.3f},{y-3:.3f}l6,6m0,-6l-6,6" stroke="#b2182b" stroke-width="1.4"/>')
   svg += [f'<text x="{x0}" y="{y0+278}" font-family="monospace" font-size="10">x:[{lo[0]:.3f}, {hi[0]:.3f}] y:[{lo[1]:.3f}, {hi[1]:.3f}]</text>']
  svg.append('</svg>');p=dest/(sid+'.svg');p.write_text('\n'.join(svg));paths.append(str(p))
 dump('figure_manifest.json',{'files':paths,'interpolated_occupancy_colored_as_success':False,'representation_surfaces_not_drawn_as_certified':True})
 print('Saved',len(paths),'observed-only scientific SVG figures')
if __name__=='__main__':main()
