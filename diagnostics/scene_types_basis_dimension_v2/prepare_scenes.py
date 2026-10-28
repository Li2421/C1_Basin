"""Generate a private, auditable v2 variant of the original scene builder."""
import guard
from guard import HERE,ROOT
import hashlib,json

def main():
    source=ROOT/'diagnostics/scene_types_basis_dimension_v1/scenes.py'
    text=source.read_text();edits=[]
    def replace(old,new):
        nonlocal text
        assert text.count(old)==1,old
        text=text.replace(old,new);edits.append(dict(old=old,new=new))
    replace('7:[12,15,18]','7:[6,8,10]')
    replace('17:[.5,.44,.4]','17:[12,16,20]')
    begin=text.index('    elif type_id==7:\n');end=text.index('    elif type_id in (8,9):',begin)
    text=text[:begin]+'''    elif type_id==7:
        from scene_helpers import forest_layout
        rects,layout_seed=forest_layout(int(v))
        notes.append(f'0.3m forest blocks, pair/wall clearance>=0.3m; geometry-only seed {layout_seed}.')
'''+text[end:]
    begin=text.index('    elif type_id==16:\n');end=text.index('    if type_id not in (1,2,6,17):',begin)
    text=text[:begin]+'''    elif type_id==16:
        n=int(v);w=.4+widen;hx=1.2+2*w;hy=1.2
        rects=[(x-.4,y-.15,x+.4,y+.15)for x in (-(.8+w),0.,.8+w)for y in (-(.15+w/2),.15+w/2)]
        notes.append('Shelf gaps and both x-end aisles equal prescribed width; outer y staging regions follow fixed half-height1.2m.')
    elif type_id==17:n=int(v)
'''+text[end:]
    begin=text.index('            elif type_id==17:\n');end=text.index('            else:\n',begin)
    text=text[:begin]+'''            elif type_id==17:
                phase=rng.uniform(0,2*np.pi)
                a=phase+np.arange(n)*2*np.pi/n+rng.uniform(-.05,.05,n)
                p=1.4*np.stack([np.cos(a),np.sin(a)],axis=1)
                rows,cols={12:(3,4),16:(4,4),20:(4,5)}[n]
                x=(np.arange(cols)-(cols-1)/2)*.44
                y=(np.arange(rows)-(rows-1)/2)*.44
                goals=np.array([[xx,yy]for yy in y for xx in x])[rng.permutation(n)]
'''+text[end:]
    replace('mintrip=1. if type_id==4 else 0.','mintrip=1.5 if type_id==7 else 1. if type_id==4 else 0.')
    replace('math.ceil(4*longest/.5/.05)','math.ceil(6*longest/.5/.05)')
    target=HERE/'generated/scenes_v2_impl.py';target.write_text(text)
    (HERE/'SCENE_BUILDER_INPUTS.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),target=str(target),target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),literal_replacements=edits,block_changes=['forest layout','uniform warehouse x-end/internal aisle widths','circle-to-compact-grid gathering']),indent=2)+'\n')
    (HERE/'SCENE2_RESOLUTION.json').write_text(json.dumps(dict(radius=.05,half_extent=1.5,source='User v2 prompt explicitly resolves the original longitudinal queue geometry.'),indent=2)+'\n')
    print('Private scene builder ready')
if __name__=='__main__':main()
