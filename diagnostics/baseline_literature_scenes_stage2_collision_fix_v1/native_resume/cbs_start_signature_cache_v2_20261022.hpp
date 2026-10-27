// Candidate only. Exact input equivalence for the unchanged constrained A*.
// With nr==0, continuous start coordinates occur only in time-zero edge
// predicates and in output[0]. The rounded start vertex and all those predicate
// truth values therefore determine the unchanged search. Output[0] is refreshed.
#include <string>
#include <cfenv>

class StartSignatureMemo {
    std::unordered_map<std::string,std::vector<double>> entries_;
    std::vector<double> context_;
    bool ready_=false;
    size_t bytes_=0;
    template<class T> static void append(std::string& key,const T& value){
        key.append(reinterpret_cast<const char*>(&value),sizeof(value));
    }
public:
    uint64_t lookups=0,hits=0,negative_hits=0,clears=0;
    void prepare(double res,double inner,double outer,double sep,double margin,
                 const double* bounds,int nw,const double* walls){
        std::vector<double> next;
        next.reserve(9+size_t(nw)*4);
        for(double x:{res,inner,outer,sep,margin,bounds[0],bounds[1],bounds[2],bounds[3]})next.push_back(x);
        if(nw)next.insert(next.end(),walls,walls+size_t(nw)*4);
        if(!ready_ || context_.size()!=next.size() ||
           std::memcmp(context_.data(),next.data(),next.size()*sizeof(double))!=0){
            entries_.clear();bytes_=0;context_.swap(next);ready_=true;++clears;
        }
    }
    std::string key(int sx,int sy,int max_time,const double* goal,int nc,
                    const double* constraints,uint16_t signature){
        std::string value;
        value.reserve(38+size_t(nc)*5*sizeof(double));
        append(value,sx);append(value,sy);append(value,max_time);
        value.append(reinterpret_cast<const char*>(goal),2*sizeof(double));
        // Time-zero exclusions affect only the ten predicates in signature.
        // Every later edge ignores them, so their coordinates/count are not
        // independent inputs to the search once that signature is fixed.
        int later_count=0;
        for(int c=0;c<nc;++c)if(int(constraints[5*c])!=0)++later_count;
        append(value,later_count);append(value,signature);
        int rounding_mode=std::fegetround();append(value,rounding_mode);
        for(int c=0;c<nc;++c)if(int(constraints[5*c])!=0)
            value.append(reinterpret_cast<const char*>(constraints+5*c),5*sizeof(double));
        return value;
    }
    bool get(const std::string& key,const double* start,double* output,int& count){
        ++lookups;
        auto it=entries_.find(key);
        if(it==entries_.end())return false;
        ++hits;count=int(it->second.size()/2);
        if(count){
            const double x=start[0],y=start[1];
            std::memcpy(output,it->second.data(),it->second.size()*sizeof(double));
            output[0]=x;output[1]=y;
        }else ++negative_hits;
        return true;
    }
    void put(const std::string& key,const double* output,int count){
        const size_t amount=key.size()+size_t(count)*2*sizeof(double);
        if(entries_.size()>=8192 || bytes_+amount>32u*1024u*1024u){
            entries_.clear();bytes_=0;++clears;
        }
        std::vector<double> path;
        if(count)path.assign(output,output+size_t(count)*2);
        if(entries_.emplace(key,std::move(path)).second)bytes_+=amount;
    }
    void reset(){entries_.clear();context_.clear();ready_=false;bytes_=0;lookups=hits=negative_hits=clears=0;}
    size_t size()const{return entries_.size();}
};
static StartSignatureMemo start_signature_memo;
