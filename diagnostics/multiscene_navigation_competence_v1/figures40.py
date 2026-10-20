"""Standalone scientific SVG figures from frozen numeric evidence only."""
from common import *
from deliver_failed_expert_pilot import NAMES
import html

def fewshot_figure(folder,colors):
    source=HERE/'AF_ANALYSIS40_RESULTS.json'
    if not source.exists():return None
    result=read(source);families=result['families'];entries=[curve for a in families.values() for curve in a.get('fewshot_curves',{}).values() if curve['mean_regret_pp'] is not None]
    ymax=max(10,int(np.ceil(max([r['mean_regret_pp'] for r in entries] or [10])/10))*10)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="540" viewBox="0 0 1120 540">','<rect width="1120" height="540" fill="white"/>','<text x="60" y="30" font-size="21">少量预实验标签选择：完整菜单平均遗憾</text>']
    for panel,(key,title) in enumerate([('empirical','经验选择'),('bayesian','贝叶斯选择')]):
        ox=65+panel*550;x=lambda k:ox+95*np.log2(k);y=lambda v:415-310*v/ymax
        svg.extend([f'<text x="{ox}" y="70" font-size="18">{title}</text>',f'<path d="M{ox} 95 V415 H{ox+380}" fill="none" stroke="#333"/>'])
        for value in np.linspace(0,ymax,6):svg.extend([f'<path d="M{ox} {y(value)} H{ox+380}" stroke="#ddd"/>',f'<text x="{ox-38}" y="{y(value)+5}" font-size="13">{value:.0f}</text>'])
        for k in [1,2,4,8,16]:svg.append(f'<text x="{x(k)-5}" y="440" font-size="14">{k}</text>')
        svg.append(f'<text x="{ox+90}" y="466" font-size="15">每个修正臂的已知标签条数</text>')
        for f,a in families.items():
            points=[]
            for k in [1,2,4,8,16]:
                curve=a.get('fewshot_curves',{}).get(f'{key}_{k}')
                if not curve or curve['mean_regret_pp'] is None:continue
                px,py=x(k),y(curve['mean_regret_pp']);points.append((px,py))
                svg.append(f'<circle cx="{px}" cy="{py}" r="4" fill="{colors[f] if curve["all_controllers_available"] else "white"}" stroke="{colors[f]}" stroke-width="2"/>')
            if points:svg.append('<polyline points="'+' '.join(f'{px},{py}' for px,py in points)+f'" fill="none" stroke="{colors[f]}" stroke-width="1.7"/>')
    for i,f in enumerate('ABCDEF'):
        ox=60+i*175;svg.extend([f'<rect x="{ox}" y="490" width="11" height="11" fill="{colors[f]}"/>',f'<text x="{ox+17}" y="501" font-size="13">{html.escape(NAMES[f])}</text>'])
    svg.append('<text x="60" y="526" font-size="13">无数据条件不补零；空心点表示实例覆盖不齐。均值基于冻结的一千次标签抽样；总预算为横轴的七倍。</text></svg>')
    output=folder/'fewshot_labels_regret.svg';output.write_text('\n'.join(svg))
    return dict(source_path=str(source),source_sha256=sha(source),output_path=str(output),output_sha256=sha(output),points=sum(1 for r in entries),partial_controller_points=sum(not r['all_controllers_available'] for r in entries))

def main():
    p=HERE/'SCALE_CURVE40_RESULTS.json'
    if not p.exists():return
    result=read(p);rows=result['rows'];folder=HERE/'figures';folder.mkdir(exist_ok=True)
    colors=dict(zip('ABCDEF',['#0072b2','#d55e00','#009e73','#cc79a7','#e69f00','#56b4e9']))
    ymax=max(10,int(np.ceil(max([r['regret_pp'] for r in rows] or [10])/10))*10);parts=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="560" viewBox="0 0 900 560">','<rect width="900" height="560" fill="white"/>','<text x="70" y="30" font-size="20">训练族数量与完整菜单遗憾</text>']
    x=lambda m:80+120*m;y=lambda v:450-370*v/ymax
    parts+=['<path d="M80 65 V450 H710" fill="none" stroke="#333"/>','<text x="310" y="505" font-size="16">新的训练族数</text>','<text x="12" y="50" font-size="15">遗憾（百分点）</text>']
    for m in [0,1,3,5]:parts.append(f'<text x="{x(m)-5}" y="475" font-size="14">{m}</text>')
    for value in np.linspace(0,ymax,6):parts.extend([f'<path d="M80 {y(value)} H710" stroke="#ddd"/>',f'<text x="35" y="{y(value)+5}" font-size="13">{value:.0f}</text>'])
    for f in 'ABCDEF':
        group=[r for r in rows if r['family']==f];means=[]
        for m in [0,1,3,5]:
            values=[r['regret_pp'] for r in group if r['m']==m]
            if values:means.append((x(m),y(np.mean(values))))
        if means:parts.append('<polyline points="'+' '.join(f'{a},{b}' for a,b in means)+f'" fill="none" stroke="{colors[f]}" stroke-width="2"/>')
        for r in group:parts.append(f'<circle cx="{x(r["m"])}" cy="{y(r["regret_pp"])}" r="4" fill="{colors[f]}" fill-opacity=".6"/>')
        parts.extend([f'<rect x="745" y="{85+30*"ABCDEF".index(f)}" width="12" height="12" fill="{colors[f]}"/>',f'<text x="765" y="{96+30*"ABCDEF".index(f)}" font-size="13">{html.escape(NAMES[f])}</text>'])
    parts.append('<text x="75" y="540" font-size="13">缺失条件不画为零；点为固定组合，线为该族可得组合均值。完整性缺口见报告。</text></svg>')
    (folder/'training_families_regret.svg').write_text('\n'.join(parts))
    fewshot=fewshot_figure(folder,colors)
    write(HERE/'FIGURES40_PROVENANCE.json',dict(utc=utc(),source_path=str(p),source_sha256=sha(p),source_rows=len(rows),fewshot=fewshot,missing_conditions_not_zero=True,controller_changes=0,physics_changes=0,success_criteria_changes=0))

if __name__=='__main__':main()
