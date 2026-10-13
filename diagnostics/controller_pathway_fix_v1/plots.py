"""Standalone scientific figures from saved endpoint/diagnostic results."""
from util import *
os.environ['MPLCONFIGDIR']=str(HERE/'matplotlib_cache')
sys.path.insert(0,str(R4/'plot_vendor'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    out=HERE/'figures';out.mkdir(exist_ok=True)
    for bed in ('blend','real'):
        f=HERE/(bed+'_results.json')
        if not f.exists():continue
        r=read(f);coverage=sorted({s['coverage'] for s in r['summaries']});n=max(sum(s['coverage']==c for s in r['summaries']) for c in coverage)
        for metric in ('E1','E2','E3'):
            fig,axes=plt.subplots(1,len(coverage),figsize=(8*len(coverage),max(7,.29*n)),squeeze=False)
            for ax,c in zip(axes[0],coverage):
                ss=[s for s in r['summaries'] if s['coverage']==c];labels=[s['method'] for s in ss];y=np.arange(len(ss))
                if metric=='E1':
                    points=[100*s['E1']['mean_lower'] for s in ss];interval=np.array([s['E1']['ci95_simultaneous'] for s in ss])*100
                    ax.hlines(y,interval[:,0],interval[:,1],color='#566b8a');ax.scatter(points,y,c=['#18846b' if s['E1']['passed'] else '#bd4b42' for s in ss],s=24);ax.axvline(-2,color='#bd4b42',ls='--',label='Noninferiority margin -2 pp');ax.set_xlabel('Success difference vs frozen R0 (pp), simultaneous 95% CI')
                elif metric=='E2':
                    for key,offset,color,label in [('vs_src',-.12,'#2875b9','Source prior'),('vs_R0',.12,'#a96222','R0')]:
                        point=np.array([s['E2'][key]['lower'] for s in ss])*100;interval=np.array([s['E2'][key]['ci95_descriptive'] for s in ss])*100
                        ax.hlines(y+offset,interval[:,0],interval[:,1],color=color);ax.scatter(point,y+offset,c=color,s=16,label=label)
                    ax.set_xlabel('Necessary-controller success gain (pp), descriptive 95% CI')
                else:
                    certain=np.array([s['E3']['certain_count']/s['controller_count'] for s in ss])*100;possible=np.array([s['E3']['possible_rate'] for s in ss])*100
                    ax.barh(y,possible,color='#efb2aa',label='Possible harm incl. unknown bounds');ax.barh(y,certain,color='#bd4b42',label='Certain harm');ax.set_xlabel('Controllers losing >=10 pp vs source prior (%)');ax.set_xlim(0,100)
                ax.axvline(0,color='#777',lw=.8);ax.set_yticks(y,labels,fontsize=7);ax.invert_yaxis();ax.set_title(f'{bed.upper()} / {c} / {metric} — DEVELOPMENT');ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8,loc='lower right')
            fig.tight_layout()
            for ext in ('png','pdf'):fig.savefig(out/f'{bed}_{metric}.{ext}',dpi=170,bbox_inches='tight')
            plt.close(fig)
        for diag in ('D1','D2','D3'):
            fig,axes=plt.subplots(1,len(coverage),figsize=(7*len(coverage),5),squeeze=False)
            for ax,c in zip(axes[0],coverage):
                if diag=='D1':
                    for arm in sorted({d['arm'] for d in r['D1']}):
                        ds=[d for d in r['D1'] if d['coverage']==c and d['arm']==arm and d['policy']=='argmax' and d.get('convex_hull') is not None]
                        ax.scatter([d['convex_hull'] for d in ds],[100*d['relative_R0_loss_upper'] for d in ds],s=24,label=arm,alpha=.7)
                    ax.set_xlabel('Source-normalized distance to controller convex hull');ax.set_ylabel('Success loss vs R0 (pp, conservative bound)')
                elif diag=='D2':
                    for arm in sorted({d['arm'] for d in r['rows']}):
                        ds=[d for d in r['rows'] if d['coverage']==c and d['arm']==arm and d['policy']=='argmax' and 'mean_predicted' in d['D2']]
                        ax.scatter([d['D2']['observed_valid_Q'] for d in ds],[d['D2']['mean_predicted'] for d in ds],s=24,label=arm,alpha=.7)
                    ax.plot([0,1],[0,1],ls='--',color='#555');ax.set_xlabel('Selected eta endpoint valid success');ax.set_ylabel('Predicted selected eta Q');ax.set_xlim(0,1);ax.set_ylim(0,1)
                else:
                    ds=[d for d in r['D3'] if d['coverage']==c and d['method'].endswith('__argmax')]
                    methods=sorted({d['method'] for d in ds});labels=[];v=[]
                    for method in methods:
                        rows=[d for d in ds if d['method']==method];hist=np.mean([[h['selection_probability'] for h in d['heads']] for d in rows],0);source=np.mean([[h['equal_controller_source_Q'] for h in d['heads']] for d in rows],0)
                        labels.append(method.split('__')[0]);v.append([hist[1],hist[8],float(hist@(source<.2))])
                    v=np.array(v)*100;x=np.arange(len(labels));ax.bar(x-.23,v[:,0],width=.23,label='Select eta1');ax.bar(x,v[:,1],width=.23,label='Select eta8');ax.bar(x+.23,v[:,2],width=.23,label='Select source-Q < .2');ax.set_xticks(x,labels,rotation=45,ha='right',fontsize=8);ax.set_ylabel('Selection frequency (%)')
                ax.set_title(f'{bed.upper()} / {c} / {diag} — DEVELOPMENT');ax.grid(alpha=.2);ax.legend(fontsize=7)
            fig.tight_layout()
            for ext in ('png','pdf'):fig.savefig(out/f'{bed}_{diag}.{ext}',dpi=170,bbox_inches='tight')
            plt.close(fig)
    write(HERE/'FIGURE_INDEX.json',[dict(path=str(f),sha256=sha(f)) for f in sorted(out.glob('*')) if f.suffix in ('.png','.pdf')])
if __name__=='__main__':main()
