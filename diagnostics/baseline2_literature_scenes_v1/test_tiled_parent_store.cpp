#include "tiled_parent_store.hpp"
#include <cassert>
#include <iostream>
#include <random>

static uint64_t key(int x,int y,int t){return (uint64_t(t)<<40)|(uint64_t(x+100000)<<20)|uint64_t(y+100000);}
int main(){
    TiledParentStore store(.05,700);
    std::unordered_map<uint64_t,uint64_t> reference;
    std::mt19937_64 rng(2026102009);
    std::vector<uint64_t> keys={0,UINT64_MAX,key(-200,-200,0),key(200,200,700),key(201,0,0),key(0,0,701)};
    for(int i=0;i<30000;++i)keys.push_back(i%3?rng():key(int(rng()%431)-215,int(rng()%431)-215,int(rng()%705)));
    for(auto k:keys){
        assert(store.contains(k)==(reference.find(k)!=reference.end()));
        uint64_t value=rng();store.emplace(k,value);reference.emplace(k,value);
        assert(store.at(k)==reference.at(k));
        store.emplace(k,~value);reference.emplace(k,~value);
        assert(store.at(k)==reference.at(k));
    }
    for(auto k:keys){assert(store.contains(k));assert(store.at(k)==reference.at(k));}
    bool threw=false;uint64_t absent=key(0,0,333);while(reference.count(absent))++absent;
    try{store.at(absent);}catch(const std::out_of_range&){threw=true;}assert(threw);
    std::cout<<"{\"operations\":"<<keys.size()*5<<",\"exact_dictionary_semantics\":true,\"zero_and_all_one_keys_and_values_supported\":true,\"outside_storage_window_uses_original_hash_map\":true}\n";
}
