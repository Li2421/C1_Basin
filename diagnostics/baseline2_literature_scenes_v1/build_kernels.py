from experiment import *
import difflib
import subprocess

GEOMETRY='''
static double cross(double ax,double ay,double bx,double by){return ax*by-ay*bx;}
static double segseg(double ax,double ay,double bx,double by,double cx,double cy,double dx,double dy){
 double ux=bx-ax,uy=by-ay,vx=dx-cx,vy=dy-cy,den=cross(ux,uy,vx,vy);
 if(std::abs(den)>1e-15){double t=cross(cx-ax,cy-ay,vx,vy)/den,s=cross(cx-ax,cy-ay,ux,uy)/den;if(t>=0&&t<=1&&s>=0&&s<=1)return 0.;}
 return std::min(std::min(segment(ax-cx,ay-cy,bx-cx,by-cy),segment(ax-dx,ay-dy,bx-dx,by-dy)),std::min(segment(cx-ax,cy-ay,dx-ax,dy-ay),segment(cx-bx,cy-by,dx-bx,dy-by)));
}
'''

def main():
    records=[]
    for name in ('astar_kernel','cbs_kernel'):
        source=ROOT/'diagnostics/crosstype6_shared_correction_controller_information_v1'/f'{name}.cpp';old=source.read_text();text=old
        text=text.replace('extern "C"',GEOMETRY+'\nextern "C"',1)
        text=text.replace('double* output) {','int nw,const double* walls,double margin,double* output) {')
        radial='if(segment(ax,ay,bx,by)<inner-1e-12||std::max(norm(ax,ay),norm(bx,by))>outer+1e-12)return false;'
        assert radial in text
        text=text.replace(radial,'for(int w=0;w<nw;++w){const double* z=walls+4*w;if(segseg(ax,ay,bx,by,z[0],z[1],z[2],z[3])<margin-1e-12)return false;}')
        path=HERE/f'{name}.cpp';path.write_text(text)
        patch=HERE/f'{name}_GEOMETRY.diff';patch.write_text(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile=str(source),tofile=str(path))))
        subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(path),'-o',str(HERE/f'{name}.so')],check=True)
        records.append(dict(source=str(source),source_sha256=sha(source),output_sha256=sha(path),patch_sha256=sha(patch),binary_sha256=sha(HERE/f'{name}.so'),scope='wall geometry only; search order, heuristics, reservation, constraints and budgets unchanged'))
    write(HERE/'KERNEL_PORTING.json',records)
if __name__=='__main__':main()
