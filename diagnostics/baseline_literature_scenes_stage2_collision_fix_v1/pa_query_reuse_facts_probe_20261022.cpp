// Read-only query-key probe: all planning still runs in the original .so.
#define astar pa_probe_reference_astar
#include "astar_kernel_tiled.cpp"
#undef astar
#include <cfenv>
#include <xmmintrin.h>
extern "C" int pa_query_facts(double res,double sep,const double* start,const double* goal,
 int nr,const int* lens,const double* reserved,int nw,const double* walls,double margin,long long* facts){
 prepare_memo(nw,walls,margin);
 std::vector<int> offset(nr);int total=0;
 for(int i=0;i<nr;++i){offset[i]=total;total+=lens[i];}
 auto edge=[&](double ax,double ay,double bx,double by,int t){
  if(!memo_geometry(ax,ay,bx,by,nw,walls,margin))return false;
  for(int i=0;i<nr;++i){int c=offset[i]+std::min(t,lens[i]-1),d=offset[i]+std::min(t+1,lens[i]-1);
   if(segment(ax-reserved[2*c],ay-reserved[2*c+1],bx-reserved[2*d],by-reserved[2*d+1])<sep-1e-12)return false;}
  return true;
 };
 int sx=std::nearbyint(start[0]/res),sy=std::nearbyint(start[1]/res);
 int gx=std::nearbyint(goal[0]/res),gy=std::nearbyint(goal[1]/res);
 facts[0]=sx;facts[1]=sy;facts[2]=std::fegetround();facts[3]=_mm_getcsr()&0xffc0;
 unsigned short control;asm volatile("fnstcw %0":"=m"(control));facts[4]=control;
 const int moves[9][2]={{1,0},{0,1},{-1,0},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1},{0,0}};
 int mask=0;
 for(int i=0;i<9;++i)if(edge(start[0],start[1],(sx+moves[i][0])*res,(sy+moves[i][1])*res,0))mask|=1<<i;
 if(sx==gx&&sy==gy&&edge(start[0],start[1],goal[0],goal[1],0))mask|=1<<9;
 return mask;
}
