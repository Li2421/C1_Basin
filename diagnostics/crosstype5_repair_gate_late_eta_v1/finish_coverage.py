"""Finish the already-computed coverage evidence without matplotlib or new simulations."""
from common import *
from html import escape

def main():
    assert not (HERE/'COVERAGE_STATISTICS.receipt.json').exists()
    existing={n:sha(HERE/n) for n in ('COVERAGE_RESULTS.json','COVERAGE_POINT_SUBSETS.npz','COVERAGE.md')}
    with np.load(HERE/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:
        labels=z['labels'][:,:,34:40,0];eta=z['action'][:,:,34:40,3:6];frozen=z['proposals']
    base=read(HERE/'BASE_PROTOCOL.json');body=['<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="480" viewBox="0 0 1320 480"><rect width="1320" height="480" fill="white"/><g font-family="sans-serif" fill="#182839">','<text x="35" y="32" font-size="20">Late eta in exploration labels: development evidence, 19 controller types</text>'];histograms={}
    for j,(low,high) in enumerate(((.5,1.25),(-.5,.5),(0,.75))):
        x0=55+j*435;y0=345;width=360;height=245;edges=np.linspace(low,high,36);series=[]
        for outcome,color,label in ((0,'#b87d77','Known failure'),(1,'#218575','Success')):
            values=eta[...,j][labels==outcome];counts,_=np.histogram(values,bins=edges);density=counts/max(len(values),1)/np.diff(edges);series.append((color,label,len(values),density))
        ymax=max(float(d.max()) for _,_,_,d in series)*1.08 or 1
        xx=lambda v:x0+(float(v)-low)/(high-low)*width
        a,b=float(frozen[...,j].min()),float(frozen[...,j].max());body.append(f'<rect x="{xx(a):.3f}" y="{y0-height}" width="{xx(b)-xx(a):.3f}" height="{height}" fill="#4566ad" opacity=".13"/>')
        for color,label,n,density in series:
            for k,d in enumerate(density):body.append(f'<rect x="{xx(edges[k]):.3f}" y="{y0-float(d)/ymax*height:.3f}" width="{width/35:.3f}" height="{float(d)/ymax*height:.3f}" fill="{color}" opacity=".45"/>')
            body.append(f'<text x="{x0}" y="{385+(0 if label=="Known failure" else 20)}" font-size="12" fill="{color}">{escape(label)} (n={n})</text>')
        for anchor,color,dash in ((13,'#7b3fa0','6,4'),(10,'#c08016','2,4')):
            x=xx(base['eta'][anchor][j]);body.append(f'<path d="M{x:.3f} {y0-height} V{y0}" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/><text x="{x:.3f}" y="{y0-height-9}" fill="{color}" font-size="12">eta{anchor}</text>')
        body.append(f'<path d="M{x0} {y0-height} V{y0} H{x0+width}" stroke="#182839" fill="none"/><text x="{x0+width/2}" y="370" text-anchor="middle">Late eta coordinate {j+1}</text>')
        for q in range(5):
            x=low+(high-low)*q/4;body.append(f'<text x="{xx(x):.3f}" y="{y0+15}" font-size="10" text-anchor="middle">{x:.3g}</text>');d=ymax*q/4;y=y0-height*q/4;body.append(f'<text x="{x0-5}" y="{y+4}" font-size="10" text-anchor="end">{d:.2g}</text>')
        histograms[str(j+1)]=dict(edges=edges,series=[dict(label=l,n=n,density=d) for _,l,n,d in series],frozen_range=[a,b],reference_eta13=base['eta'][13][j],reference_eta10=base['eta'][10][j])
    body+=['<text x="35" y="452" font-size="12">Density histograms; blue band: frozen proposals range. eta13/eta10 are reference markers only, never training targets.</text>','</g></svg>'];path=HERE/'figures/late_eta_distribution.svg';path.parent.mkdir(exist_ok=True);path.write_text(''.join(body))
    write(HERE/'EXPLORATION_HISTOGRAMS.json',histograms)
    assert all(sha(HERE/n)==h for n,h in existing.items())
    seal('COVERAGE_STATISTICS.receipt.json',[HERE/n for n in ('COVERAGE_RESULTS.json','COVERAGE_POINT_SUBSETS.npz','COVERAGE.md','figures/late_eta_distribution.svg','EXPLORATION_HISTOGRAMS.json','coverage_statistics.py','finish_coverage.py','COVERAGE_BUNDLE.receipt.json','COVERAGE_BOOTSTRAP_IMPLEMENTATION.md')],saved_bootstrap_reused_without_modification=True,no_new_rollouts=True,no_package_installation=True)
if __name__=='__main__':main()
