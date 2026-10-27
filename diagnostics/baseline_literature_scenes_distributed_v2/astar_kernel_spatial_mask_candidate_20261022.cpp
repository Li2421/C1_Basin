#include "tiled_parent_store.hpp"
// Exact search order of SpaceTimeGrid, with scalar double geometry.
#include <cmath>
#include <vector>
#include <queue>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
struct Node { double f; uint64_t serial; int x,y,t; };
struct Greater { bool operator()(const Node&a,const Node&b) const {return a.f!=b.f ? a.f>b.f : a.serial>b.serial;} };
static uint64_t code(int x,int y,int t) {return (uint64_t(t)<<40)|(uint64_t(x+100000)<<20)|uint64_t(y+100000);}
static double norm(double x,double y) {return std::sqrt(x*x+y*y);}
static double segment(double ax,double ay,double bx,double by) {double dx=bx-ax,dy=by-ay;double den=std::max(dx*dx+dy*dy,1e-30);double t=std::min(1.,std::max(0.,-(ax*dx+ay*dy)/den));return norm(ax+t*dx,ay+t*dy);}

static double cross(double ax,double ay,double bx,double by){return ax*by-ay*bx;}
static double segseg(double ax,double ay,double bx,double by,double cx,double cy,double dx,double dy){
 double ux=bx-ax,uy=by-ay,vx=dx-cx,vy=dy-cy,den=cross(ux,uy,vx,vy);
 if(std::abs(den)>1e-15){double t=cross(cx-ax,cy-ay,vx,vy)/den,s=cross(cx-ax,cy-ay,ux,uy)/den;if(t>=0&&t<=1&&s>=0&&s<=1)return 0.;}
 return std::min(std::min(segment(ax-cx,ay-cy,bx-cx,by-cy),segment(ax-dx,ay-dy,bx-dx,by-dy)),std::min(segment(cx-ax,cy-ay,dx-ax,dy-ay),segment(cx-bx,cy-by,dx-bx,dy-by)));
}


#include <array>
#include <cstring>
struct EdgeKey { std::array<uint64_t,4> b; bool operator==(const EdgeKey&o)const{return b==o.b;} };
struct EdgeHash {size_t operator()(const EdgeKey&k)const{size_t h=0;for(auto b:k.b)h^=std::hash<uint64_t>{}(b)+0x9e3779b9+(h<<6)+(h>>2);return h;}};
static std::vector<double> memo_walls;
static double memo_margin=0;
static std::unordered_map<EdgeKey,bool,EdgeHash> memo;
static bool ready=false;
static void prepare_memo(int nw,const double* walls,double margin){
 if(!ready||memo_walls.size()!=size_t(4*nw)||std::memcmp(&margin,&memo_margin,sizeof(double))!=0||(nw&&std::memcmp(memo_walls.data(),walls,4*nw*sizeof(double))!=0)){
  memo_walls.assign(walls,walls+4*nw);memo_margin=margin;memo.clear();ready=true;
 }
}
static bool memo_geometry(double ax,double ay,double bx,double by,int nw,const double* walls,double margin){
 if(nw==0)return true;
 EdgeKey key;double values[4]={ax,ay,bx,by};std::memcpy(key.b.data(),values,sizeof(values));
 auto found=memo.find(key);if(found!=memo.end())return found->second;
 bool valid=true;
 for(int w=0;w<nw;++w){const double* z=walls+4*w;if(segseg(ax,ay,bx,by,z[0],z[1],z[2],z[3])<margin-1e-12){valid=false;break;}}
 if(memo.size()>1000000)memo.clear();
 memo.emplace(key,valid);return valid;
}

#include <new>
static uint64_t mask_stats[8]={};
extern "C" void pa_stationary_edge_stats(uint64_t* out){std::copy(mask_stats,mask_stats+8,out);}
struct CellMasks {uint16_t wall_known=0,wall_forbidden=0;std::array<uint16_t,16> known{},forbidden{};};
extern "C" int astar(double res,double inner,double outer,double sep,int max_time,const double* start,const double* goal,int nr,const int* lens,const double* reserved,int nw,const double* walls,double margin,double* output) {
 prepare_memo(nw,walls,margin);
 std::vector<int> offset(nr);int total=0;for(int i=0;i<nr;++i){offset[i]=total;total+=lens[i];}
 ++mask_stats[0];std::unordered_map<uint64_t,CellMasks> spatial_cells;
 auto edge=[&](double ax,double ay,double bx,double by,int t,bool grid_edge=false,uint64_t cell=0,int move=0){
  CellMasks* cache=nullptr;const uint16_t bit=uint16_t(1u<<move);
  if(grid_edge&&t>0&&nr<=16){
   ++mask_stats[1];auto found=spatial_cells.find(cell);
   if(found!=spatial_cells.end())cache=&found->second;
   else if(spatial_cells.size()<65536){try{cache=&spatial_cells.emplace(cell,CellMasks{}).first->second;++mask_stats[4];mask_stats[7]=std::max(mask_stats[7],uint64_t(spatial_cells.size()));}catch(const std::bad_alloc&){++mask_stats[6];}}
   else ++mask_stats[5];
  }
  bool wall_valid;
  if(cache&&(cache->wall_known&bit)){++mask_stats[2];wall_valid=!(cache->wall_forbidden&bit);}
  else{wall_valid=memo_geometry(ax,ay,bx,by,nw,walls,margin);if(cache){cache->wall_known|=bit;if(!wall_valid)cache->wall_forbidden|=bit;}}
  if(!wall_valid)return false;
  for(int i=0;i<nr;++i){
   const bool stationary=cache&&t>=lens[i]-1;
   if(stationary&&(cache->known[i]&bit)){++mask_stats[2];if(cache->forbidden[i]&bit)return false;continue;}
   int c=offset[i]+std::min(t,lens[i]-1),d=offset[i]+std::min(t+1,lens[i]-1);++mask_stats[3];
   const bool forbidden=segment(ax-reserved[2*c],ay-reserved[2*c+1],bx-reserved[2*d],by-reserved[2*d+1])<sep-1e-12;
   if(stationary){cache->known[i]|=bit;if(forbidden)cache->forbidden[i]|=bit;}
   if(forbidden)return false;
  }
  return true;
 };
 int sx=std::nearbyint(start[0]/res),sy=std::nearbyint(start[1]/res),gx=std::nearbyint(goal[0]/res),gy=std::nearbyint(goal[1]/res);

 // Exact no-incoming-edge certificate, using the original endpoint term.
 // No goal, resolution, radius, search tie, reservation or budget is changed.
 if(sx!=gx||sy!=gy){
  const double bx=gx*res,by=gy*res;
  for(int w=0;w<nw;++w){const double* z=walls+4*w;
   if(segment(z[0]-bx,z[1]-by,z[2]-bx,z[3]-by)<margin-1e-12)return 0;
  }
 }
 auto heur=[&](int x,int y){return norm(double(x-gx),double(y-gy))/std::sqrt(2.);};
 std::priority_queue<Node,std::vector<Node>,Greater> q;TiledParentStore parent(res,max_time);uint64_t serial=0;const auto initial=code(sx,sy,0);parent.emplace(initial,initial);q.push({heur(sx,sy),0,sx,sy,0});
 const int moves[9][2]={{1,0},{0,1},{-1,0},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1},{0,0}};
 while(!q.empty()){
  Node u=q.top();q.pop();double ax=u.t==0?start[0]:u.x*res,ay=u.t==0?start[1]:u.y*res;
  if(u.x==gx&&u.y==gy&&edge(ax,ay,goal[0],goal[1],u.t)){
   bool held=true;for(int t=u.t+1;t<max_time;++t)if(!edge(goal[0],goal[1],goal[0],goal[1],t)){held=false;break;}
   if(held){std::vector<uint64_t> chain;uint64_t id=code(u.x,u.y,u.t);chain.push_back(id);while(id!=initial){id=parent.at(id);chain.push_back(id);}std::reverse(chain.begin(),chain.end());int k=0;output[k++]=start[0];output[k++]=start[1];for(size_t j=1;j<chain.size();++j){int x=int((chain[j]>>20)&((uint64_t(1)<<20)-1))-100000,y=int(chain[j]&((uint64_t(1)<<20)-1))-100000;output[k++]=x*res;output[k++]=y*res;}output[k++]=goal[0];output[k++]=goal[1];return k/2;}
  }
  if(u.t>=max_time-1)continue;
  for(int mi=0;mi<9;++mi){const auto&m=moves[mi];int x=u.x+m[0],y=u.y+m[1],t=u.t+1;uint64_t id=code(x,y,t);if(parent.contains(id))continue;if(!edge(ax,ay,x*res,y*res,u.t,true,code(u.x,u.y,0),mi))continue;parent.emplace(id,code(u.x,u.y,u.t));q.push({double(t)+heur(x,y),++serial,x,y,t});}
 }
 return 0;
}
