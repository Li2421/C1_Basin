// Exact dictionary for the ORIGINAL packed uint64_t keys. Storage only:
// no node, edge, queue order, heuristic, budget, or search state is pruned.
#pragma once
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

class TiledParentStore {
    struct Tile {
        uint8_t seen[1024]{};
        uint64_t values[1024];
    };
    int extent_=0, width_=0, tiles_=0, times_=0;
    std::vector<std::unique_ptr<Tile>> pages_;
    std::unordered_map<uint64_t,uint64_t> other_;
    bool location(uint64_t key, size_t &page, size_t &cell) const {
        const uint64_t t=key>>40;
        const int x=int((key>>20)&((uint64_t(1)<<20)-1))-100000+extent_;
        const int y=int(key&((uint64_t(1)<<20)-1))-100000+extent_;
        if(!tiles_ || t>=uint64_t(times_) || x<0 || y<0 || x>=width_ || y>=width_)return false;
        page=(size_t(t)*tiles_+size_t(x>>5))*tiles_+size_t(y>>5);
        cell=size_t(x&31)*32+size_t(y&31);
        return true;
    }
public:
    TiledParentStore(double resolution,int max_time) {
        // This window is ONLY a memory layout. All other packed keys retain
        // the original unordered_map implementation, including unusual keys.
        if(!std::isfinite(resolution)||resolution<=0||max_time<0)return;
        double extent=std::ceil(10./resolution);
        if(extent>4096)return;
        extent_=int(extent);width_=2*extent_+1;tiles_=(width_+31)/32;times_=max_time+1;
        const size_t count=size_t(times_)*tiles_*tiles_;
        if(count>5000000){tiles_=0;return;}
        pages_.resize(count);
    }
    bool contains(uint64_t key) const {
        size_t p,c;
        if(!location(key,p,c))return other_.find(key)!=other_.end();
        return pages_[p] && pages_[p]->seen[c];
    }
    void emplace(uint64_t key,uint64_t value) {
        size_t p,c;
        if(!location(key,p,c)){other_.emplace(key,value);return;}
        if(!pages_[p])pages_[p].reset(new Tile);
        auto &tile=*pages_[p];
        if(!tile.seen[c]){tile.values[c]=value;tile.seen[c]=1;}
    }
    uint64_t at(uint64_t key) const {
        size_t p,c;
        if(!location(key,p,c))return other_.at(key);
        if(!pages_[p]||!pages_[p]->seen[c])throw std::out_of_range("parent key absent");
        return pages_[p]->values[c];
    }
};
