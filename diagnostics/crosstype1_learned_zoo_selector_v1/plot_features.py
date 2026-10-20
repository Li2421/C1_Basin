"""Vector plots using the repository's SVG convention; no new dependencies."""
from common import *
import csv
from html import escape

def main(cohort):
    path=HERE/f'{cohort.upper()}_FEATURES.npz'
    receipt=read(HERE/f'{cohort.upper()}_FEATURES.receipt.json')
    for source,digest in receipt['files'].items():assert sha(source)==digest
    with np.load(path,allow_pickle=False) as z:
        x=z['features'];types=z['types'];names=z['names'];features=z['feature_names']
    order=[t for t in ('Flow','gaussian','deterministic','mixture','diffusion','chunking','recurrent','cvae','ring_ippo','implicit') if t in types]
    width,height=2000,1980;panel_w,panel_h=500,315
    unavailable=int(np.sum(~np.isfinite(x).any(1)));summaries=[]
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#17202a}</style>']
    def text(px,py,value,size=12,extra=''):
        svg.append(f'<text x="{px:.2f}" y="{py:.2f}" font-size="{size}" {extra}>{escape(str(value))}</text>')
    text(30,30,f'{cohort.capitalize()} probe features: {len(names)} controllers; {unavailable} unavailable feature vectors',20)
    text(30,54,'Boxes: 25th/50th/75th percentiles; whiskers: min/max; dots: individual controllers. No task outcomes used.',13)
    for j,name in enumerate(features):
        col,row=j%4,j//4;x0=col*panel_w+65;y0=row*panel_h+105;pw=410;ph=170
        text(x0-30,y0-20,name,13)
        finite=x[np.isfinite(x[:,j]),j];low,high=(float(finite.min()),float(finite.max())) if len(finite) else (0.,1.)
        pad=max((high-low)*.08,1e-5);low-=pad;high+=pad
        y=lambda value:y0+ph-(value-low)/(high-low)*ph
        for v in np.linspace(low,high,5):
            yy=y(v);svg.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x0+pw}" y2="{yy:.2f}" stroke="#e0e4e8"/>');text(x0-8,yy+4,f'{v:.3g}',10,'text-anchor="end"')
        for i,t in enumerate(order):
            values=x[(types==t)&np.isfinite(x[:,j]),j];xx=x0+(i+.5)*pw/len(order)
            text(xx,y0+ph+18,t,10,f'transform="rotate(55 {xx:.2f} {y0+ph+18})"')
            if not len(values):continue
            minimum,q1,median,q3,maximum=np.quantile(values,[0,.25,.5,.75,1])
            summaries.append(dict(feature=str(name),type=t,n=len(values),minimum=float(minimum),q1=float(q1),median=float(median),q3=float(q3),maximum=float(maximum)))
            svg.append(f'<line x1="{xx:.2f}" y1="{y(minimum):.2f}" x2="{xx:.2f}" y2="{y(maximum):.2f}" stroke="#375c7c"/>')
            svg.append(f'<rect x="{xx-10:.2f}" y="{y(q3):.2f}" width="20" height="{max(y(q1)-y(q3),.5):.2f}" fill="#b8d6ed" stroke="#375c7c"/>')
            svg.append(f'<line x1="{xx-10:.2f}" y1="{y(median):.2f}" x2="{xx+10:.2f}" y2="{y(median):.2f}" stroke="#17202a" stroke-width="2"/>')
            for k,v in enumerate(values):
                dx=((k%5)-2)*2
                svg.append(f'<circle cx="{xx+dx:.2f}" cy="{y(v):.2f}" r="2.1" fill="#1e679c" fill-opacity=".7"/>')
    svg.append('</svg>');folder=HERE/'figures';folder.mkdir(exist_ok=True)
    (folder/f'{cohort}_feature_distributions.svg').write_text('\n'.join(svg)+'\n')
    write(folder/f'{cohort}_feature_plot_data.json',dict(input_sha256=sha(path),quantiles=summaries,unavailable_vectors=unavailable,task_outcomes_used=False))
    with (HERE/f'{cohort.upper()}_FEATURE_TABLE.csv').open('w',newline='') as out:
        writer=csv.writer(out);writer.writerow(['controller','type',*features])
        writer.writerows([str(name),str(t),*[float(v) if np.isfinite(v) else '' for v in row]] for name,t,row in zip(names,types,x))
    print('Vector feature plot and data saved',cohort,flush=True)

if __name__=='__main__':main(sys.argv[1])
