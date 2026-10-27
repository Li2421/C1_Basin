// Cache storage equivalence: exact bit keys, collisions, growth and clears.
#include "cbs_kernel_flat_cache.cpp"
#include <cassert>
#include <random>
#include <iostream>

int main(){
 FlatGeometryMemo flat;std::unordered_map<EdgeKey,bool,EdgeHash> reference;
 std::mt19937_64 rng(20261021);std::vector<EdgeKey> keys;
 for(size_t i=0;i<1100000;++i){
  EdgeKey k;for(auto& b:k.b)b=rng();keys.push_back(k);
  bool value=(rng()&1);flat.emplace(k,value);reference.emplace(k,value);
  if(i%10000==0)assert(flat.size()==reference.size());
 }
 for(const auto& key:keys){auto f=flat.find(key);assert(f!=flat.end());assert(f->second==reference.at(key));}
 for(size_t i=0;i<keys.size();i+=7){flat.emplace(keys[i],!reference.at(keys[i]));assert(flat.find(keys[i])->second==reference.at(keys[i]));}
 flat.clear();assert(flat.size()==0);
 for(const auto& key:keys)assert(flat.find(key)==flat.end());
 for(size_t round=0;round<8;++round){
  for(size_t i=round;i<keys.size();i+=31){flat.emplace(keys[i],round&1);assert(flat.find(keys[i])->second==bool(round&1));}
  flat.clear();
 }
 EdgeKey a{},b{};a.b={0,0,0,0};b.b={0x8000000000000000ULL,0,0,0};
 flat.emplace(a,true);flat.emplace(b,false);assert(flat.find(a)->second);assert(!flat.find(b)->second);
 std::cout<<"PASS: 1100000 exact bit keys, growth, first-insertion semantics, clears, signed zero distinction\n";
}
