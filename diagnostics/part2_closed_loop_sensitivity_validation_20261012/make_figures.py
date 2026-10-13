"""Small standalone SVG figures from audited CSV/JSON, with no plotting dependency."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)


def main():
    data=json.loads((HERE/'reuse_summary.json').read_text())['b15_counts']
    pairs=[('TT','TF'),('FT','FF'),('TT','FT'),('TF','FF'),('TT','FF')]
    chunks=['<svg xmlns="http://www.w3.org/2000/svg" width="850" height="390" viewBox="0 0 850 390">',
        '<rect width="850" height="390" fill="white"/>',
        '<text x="24" y="30" font-size="20" font-family="sans-serif">B15 candidate membership: paired gain and loss</text>',
        '<text x="24" y="50" font-size="12" font-family="sans-serif">24 states × 16 eta per scene; unknown pairs excluded from gain/loss bars</text>']
    colors={'gain':'#238b45','loss':'#cb181d'}
    for si,scene in enumerate(('ring_exchange','toy_give_way')):
        y0=82+si*145
        chunks.append(f'<text x="24" y="{y0}" font-size="15" font-family="sans-serif">{escape(scene)}</text>')
        for j,(old,new) in enumerate(pairs):
            x=28+j*160;g=data.get(f'{scene}:{old}:{new}:gain',0);l=data.get(f'{scene}:{old}:{new}:loss',0)
            scale=115/180; gy=y0+65-g*scale;ly=y0+65-l*scale
            chunks.append(f'<rect x="{x+25}" y="{gy:.2f}" width="35" height="{g*scale:.2f}" fill="{colors["gain"]}"/>')
            chunks.append(f'<rect x="{x+67}" y="{ly:.2f}" width="35" height="{l*scale:.2f}" fill="{colors["loss"]}"/>')
            chunks.append(f'<text x="{x+67}" y="{y0+84}" text-anchor="middle" font-size="12" font-family="sans-serif">{old}→{new}</text>')
            chunks.append(f'<text x="{x+43}" y="{gy-3:.2f}" text-anchor="middle" font-size="10" font-family="sans-serif">{g}</text>')
            chunks.append(f'<text x="{x+85}" y="{ly-3:.2f}" text-anchor="middle" font-size="10" font-family="sans-serif">{l}</text>')
    chunks.append('<rect x="24" y="358" width="12" height="12" fill="#238b45"/><text x="42" y="369" font-size="12" font-family="sans-serif">gain</text>')
    chunks.append('<rect x="98" y="358" width="12" height="12" fill="#cb181d"/><text x="116" y="369" font-size="12" font-family="sans-serif">loss</text></svg>')
    (OUT/'b15_gain_loss.svg').write_text('\n'.join(chunks)+'\n')

    rows=list(csv.DictReader((HERE/'boundary_response.csv').open()))
    chunks=['<svg xmlns="http://www.w3.org/2000/svg" width="650" height="500" viewBox="0 0 650 500">',
        '<rect width="650" height="500" fill="white"/>',
        '<text x="24" y="30" font-size="19" font-family="sans-serif">Ring local boundary response: predicted vs observed</text>',
        '<text x="24" y="50" font-size="12" font-family="sans-serif">H1/H4 registered lines; fraction coordinate; identity line shown</text>',
        '<line x1="100" y1="420" x2="560" y2="420" stroke="black"/><line x1="100" y1="420" x2="100" y2="85" stroke="black"/>']
    xmin,xmax=-.36,.36
    def xx(v):return 100+(v-xmin)/(xmax-xmin)*460
    def yy(v):return 420-(v-xmin)/(xmax-xmin)*335
    chunks.append(f'<line x1="{xx(xmin)}" y1="{yy(xmin)}" x2="{xx(xmax)}" y2="{yy(xmax)}" stroke="#999" stroke-dasharray="5 5"/>')
    for r in rows:
        actual=float(r['actual_fraction_shift']);pred=float(r['predicted_fraction_shift'])
        color='#2171b5' if r['uid'].endswith('920004') else '#ef8a62'
        chunks.append(f'<circle cx="{xx(actual):.1f}" cy="{yy(pred):.1f}" r="6" fill="{color}"/>')
        chunks.append(f'<text x="{xx(actual)+9:.1f}" y="{yy(pred)+4:.1f}" font-size="11" font-family="sans-serif">{r["uid"][-6:]} {r["source"]}→{r["target"]}</text>')
    chunks.append('<text x="330" y="468" text-anchor="middle" font-size="13" font-family="sans-serif">Observed root fraction shift</text>')
    chunks.append('<text x="18" y="250" transform="rotate(-90 18 250)" text-anchor="middle" font-size="13" font-family="sans-serif">Local linear prediction</text></svg>')
    (OUT/'boundary_response.svg').write_text('\n'.join(chunks)+'\n')


if __name__=='__main__':main()
