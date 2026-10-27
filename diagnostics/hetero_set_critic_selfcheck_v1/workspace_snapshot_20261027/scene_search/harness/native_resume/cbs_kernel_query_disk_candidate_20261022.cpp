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
// Exact bit-key -> pure geometry bool dictionary. Storage layout only.
class FlatGeometryMemo {
 struct Slot {EdgeKey first{};uint32_t generation=0;bool second=false;};
 std::vector<Slot> slots_;size_t count_=0;uint32_t generation_=1;
 static size_t hash(const EdgeKey& key){
  uint64_t h=EdgeHash{}(key);h^=h>>30;h*=0xbf58476d1ce4e5b9ULL;
  h^=h>>27;h*=0x94d049bb133111ebULL;return size_t(h^(h>>31));
 }
 void grow(){
  std::vector<Slot> old;old.swap(slots_);slots_.resize(old.empty()?32768:old.size()*2);
  count_=0;uint32_t previous=generation_;generation_=1;
  for(const auto& slot:old)if(slot.generation==previous)emplace(slot.first,slot.second);
 }
public:
 Slot* end(){return nullptr;}
 Slot* find(const EdgeKey& key){
  if(slots_.empty())return nullptr;
  size_t i=hash(key)&(slots_.size()-1);
  while(slots_[i].generation==generation_){if(slots_[i].first==key)return &slots_[i];i=(i+1)&(slots_.size()-1);}
  return nullptr;
 }
 size_t size()const{return count_;}
 void clear(){
  count_=0;
  if(++generation_==0){for(auto& slot:slots_)slot.generation=0;generation_=1;}
 }
 void emplace(const EdgeKey& key,bool value){
  if(slots_.empty()||(count_+1)*10>slots_.size()*6)grow();
  size_t i=hash(key)&(slots_.size()-1);
  while(slots_[i].generation==generation_){if(slots_[i].first==key)return;i=(i+1)&(slots_.size()-1);}
  slots_[i].first=key;slots_[i].second=value;slots_[i].generation=generation_;++count_;
 }
};
static FlatGeometryMemo memo;
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


#include "cbs_start_signature_cache_v2_20261022.hpp"
#include "cbs_query_disk_memo_candidate_20261022.hpp"

extern "C" int astar_constrained(double res,double inner,double outer,double sep,int max_time,const double* start,const double* goal,int nr,const int* lens,const double* reserved,int nc,const double* constraints,const double* bounds,int nw,const double* walls,double margin,double* output) {
 prepare_memo(nw,walls,margin);
 std::vector<int> offset(nr);int total=0;for(int i=0;i<nr;++i){offset[i]=total;total+=lens[i];}
 auto edge=[&](double ax,double ay,double bx,double by,int t){
  if(std::min(ax,bx)<bounds[0]-1e-12 || std::max(ax,bx)>bounds[1]+1e-12 || std::min(ay,by)<bounds[2]-1e-12 || std::max(ay,by)>bounds[3]+1e-12)return false;
  for(int c=0;c<nc;++c){const double* z=constraints+5*c;if(int(z[0])==t && std::abs(ax-z[1])<1e-12 && std::abs(ay-z[2])<1e-12 && std::abs(bx-z[3])<1e-12 && std::abs(by-z[4])<1e-12)return false;}
  if(!memo_geometry(ax,ay,bx,by,nw,walls,margin))return false;
  for(int i=0;i<nr;++i){int c=offset[i]+std::min(t,lens[i]-1),d=offset[i]+std::min(t+1,lens[i]-1);if(segment(ax-reserved[2*c],ay-reserved[2*c+1],bx-reserved[2*d],by-reserved[2*d+1])<sep-1e-12)return false;}return true;
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
 // BEGIN EXACT START SIGNATURE CANDIDATE
 const bool use_start_memo=(nr==0 && nc>=0 && nw>=0 && res>0 && std::isfinite(res) && std::isfinite(start[0]) && std::isfinite(start[1]) && std::isfinite(goal[0]) && std::isfinite(goal[1]));
 std::string start_memo_key;
 if(use_start_memo){
  start_signature_disk_memo.prepare(res,inner,outer,sep,margin,bounds,nw,walls);
  const int initial_moves[9][2]={{1,0},{0,1},{-1,0},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1},{0,0}};
  uint16_t signature=0;
  for(int m=0;m<9;++m)if(edge(start[0],start[1],(sx+initial_moves[m][0])*res,(sy+initial_moves[m][1])*res,0))signature|=uint16_t(1u<<m);
  if(sx==gx && sy==gy && edge(start[0],start[1],goal[0],goal[1],0))signature|=uint16_t(1u<<9);
  start_memo_key=start_signature_disk_memo.key(sx,sy,max_time,goal,nc,constraints,signature);
  int cached_count=0;
  if(start_signature_disk_memo.get(start_memo_key,start,output,cached_count))return cached_count;
 }
 // END EXACT START SIGNATURE CANDIDATE
 auto heur=[&](int x,int y){return norm(double(x-gx),double(y-gy))/std::sqrt(2.);};
 std::priority_queue<Node,std::vector<Node>,Greater> q;TiledParentStore parent(res,max_time);uint64_t serial=0;const auto initial=code(sx,sy,0);parent.emplace(initial,initial);q.push({heur(sx,sy),0,sx,sy,0});
 const int moves[9][2]={{1,0},{0,1},{-1,0},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1},{0,0}};
 while(!q.empty()){
  Node u=q.top();q.pop();double ax=u.t==0?start[0]:u.x*res,ay=u.t==0?start[1]:u.y*res;
  if(u.x==gx&&u.y==gy&&edge(ax,ay,goal[0],goal[1],u.t)){
   bool held=true;for(int t=u.t+1;t<max_time;++t)if(!edge(goal[0],goal[1],goal[0],goal[1],t)){held=false;break;}
   if(held){std::vector<uint64_t> chain;uint64_t id=code(u.x,u.y,u.t);chain.push_back(id);while(id!=initial){id=parent.at(id);chain.push_back(id);}std::reverse(chain.begin(),chain.end());int k=0;output[k++]=start[0];output[k++]=start[1];for(size_t j=1;j<chain.size();++j){int x=int((chain[j]>>20)&((uint64_t(1)<<20)-1))-100000,y=int(chain[j]&((uint64_t(1)<<20)-1))-100000;output[k++]=x*res;output[k++]=y*res;}output[k++]=goal[0];output[k++]=goal[1];if(use_start_memo)start_signature_disk_memo.put(start_memo_key,output,k/2);return k/2;}
  }
  if(u.t>=max_time-1)continue;
  for(auto&m:moves){int x=u.x+m[0],y=u.y+m[1],t=u.t+1;uint64_t id=code(x,y,t);if(parent.contains(id))continue;if(!edge(ax,ay,x*res,y*res,u.t))continue;parent.emplace(id,code(u.x,u.y,u.t));q.push({double(t)+heur(x,y),++serial,x,y,t});}
 }
 if(use_start_memo)start_signature_disk_memo.put(start_memo_key,output,0);
 return 0;
}

extern "C" void start_signature_cache_stats(uint64_t* out){out[0]=start_signature_disk_memo.lookups;out[1]=start_signature_disk_memo.hits;out[2]=start_signature_disk_memo.negative_hits;out[3]=start_signature_disk_memo.clears;out[4]=start_signature_disk_memo.size();}
extern "C" void start_signature_cache_reset(){start_signature_disk_memo.reset();}

extern "C" int query_disk_cache_configure(const char* name){return start_signature_disk_memo.disk.configure(name);}
extern "C" void query_disk_cache_enable(int yes){start_signature_disk_memo.disk.enable(yes!=0);}
extern "C" void query_disk_cache_stats(uint64_t* out){auto& d=start_signature_disk_memo.disk;out[0]=d.reads;out[1]=d.hits;out[2]=d.collisions;out[3]=d.invalid;out[4]=d.writes;out[5]=d.write_errors;out[6]=d.skipped;}
extern "C" void query_disk_cache_min_seconds(double seconds){if(seconds>=0 && std::isfinite(seconds))start_signature_disk_memo.minimum_seconds=seconds;}
