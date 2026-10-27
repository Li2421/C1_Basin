// Candidate only: bounded, verified pure-query memo, never a rollout database.
#include <openssl/sha.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

class QueryDiskMemo {
    static constexpr size_t LIMIT=16384;
    int root_=-1;
    bool enabled_=true;
    static constexpr const char* ROOT="/home/zhihan/research/Basin_C1/diagnostics/cross_scene_prospective_v1/runtime_cbs_query_cache";
    static void slot(const std::string& key,char* shard,char* file){
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(key.data()),key.size(),hash);
        std::snprintf(shard,3,"%02x",unsigned(hash[0]));
        std::snprintf(file,3,"%02x",unsigned(hash[1]));
    }
    static bool transfer(int fd,unsigned char* data,size_t n,bool writing){
        size_t done=0;
        while(done<n){
            ssize_t got=writing ? ::write(fd,data+done,n-done) : ::read(fd,data+done,n-done);
            if(got<0 && errno==EINTR)continue;
            if(got<=0)return false;
            done+=size_t(got);
        }
        return true;
    }
public:
    uint64_t reads=0,hits=0,collisions=0,invalid=0,writes=0,write_errors=0,skipped=0;
    ~QueryDiskMemo(){if(root_>=0)::close(root_);}
    int configure(const char* name){
        if(!name || std::strlen(name)!=64)return 0;
        for(int k=0;k<64;++k)if(!((name[k]>='0'&&name[k]<='9')||(name[k]>='a'&&name[k]<='f')))return 0;
        if(root_>=0){::close(root_);root_=-1;}
        if(::mkdir(ROOT,0700)<0 && errno!=EEXIST)return 0;
        int base=::open(ROOT,O_RDONLY|O_DIRECTORY|O_NOFOLLOW);
        if(base<0)return 0;
        if(::mkdirat(base,name,0700)<0 && errno!=EEXIST){::close(base);return 0;}
        root_=::openat(base,name,O_RDONLY|O_DIRECTORY|O_NOFOLLOW);::close(base);
        return root_>=0;
    }
    void enable(bool value){enabled_=value;}
    bool get(const std::string& key,int maximum,std::vector<double>& result){
        if(!enabled_ || root_<0 || key.size()+48>LIMIT)return false;
        ++reads;char shard[3],file[3];slot(key,shard,file);
        int dir=::openat(root_,shard,O_RDONLY|O_DIRECTORY|O_NOFOLLOW);
        if(dir<0)return false;
        int fd=::openat(dir,file,O_RDONLY|O_NOFOLLOW);::close(dir);
        if(fd<0)return false;
        struct stat st;
        if(::fstat(fd,&st)<0 || st.st_size<48 || uint64_t(st.st_size)>LIMIT){::close(fd);++invalid;return false;}
        std::vector<unsigned char> data(size_t(st.st_size));
        bool complete=transfer(fd,data.data(),data.size(),false);::close(fd);
        if(!complete || std::memcmp(data.data(),"BLCBSF01",8)){++invalid;return false;}
        uint32_t length=0,count=0;
        std::memcpy(&length,data.data()+8,4);std::memcpy(&count,data.data()+12,4);
        if(count>uint32_t(maximum) || size_t(length)+48+size_t(count)*16!=data.size()){++invalid;return false;}
        unsigned char hash[SHA256_DIGEST_LENGTH];SHA256(data.data(),data.size()-32,hash);
        if(std::memcmp(hash,data.data()+data.size()-32,32)){++invalid;return false;}
        if(size_t(length)!=key.size() || std::memcmp(data.data()+16,key.data(),key.size())){++collisions;return false;}
        result.resize(size_t(count)*2);
        if(count)std::memcpy(result.data(),data.data()+16+length,size_t(count)*16);
        ++hits;return true;
    }
    void put(const std::string& key,const double* value,int count){
        size_t amount=48+key.size()+size_t(count)*16;
        if(!enabled_ || root_<0 || count<0 || amount>LIMIT){++skipped;return;}
        std::vector<unsigned char> data(amount);
        std::memcpy(data.data(),"BLCBSF01",8);
        uint32_t length=uint32_t(key.size()),points=uint32_t(count);
        std::memcpy(data.data()+8,&length,4);std::memcpy(data.data()+12,&points,4);
        std::memcpy(data.data()+16,key.data(),key.size());
        if(count){
            std::memcpy(data.data()+16+length,value,size_t(count)*16);
            // Continuous start is intentionally absent from the equivalence
            // key. Its output coordinates are always restored by get().
            std::memset(data.data()+16+length,0,16);
        }
        SHA256(data.data(),data.size()-32,data.data()+data.size()-32);
        char shard[3],file[3];slot(key,shard,file);
        if(::mkdirat(root_,shard,0700)<0 && errno!=EEXIST){++write_errors;return;}
        int dir=::openat(root_,shard,O_RDONLY|O_DIRECTORY|O_NOFOLLOW);
        if(dir<0){++write_errors;return;}
        char temporary[64];std::snprintf(temporary,sizeof(temporary),".tmp.%ld",long(::getpid()));
        int fd=::openat(root_,temporary,O_WRONLY|O_CREAT|O_TRUNC|O_NOFOLLOW,0600);
        bool ok=fd>=0;
        if(ok){ok=transfer(fd,data.data(),data.size(),true) && ::fsync(fd)==0;::close(fd);}
        if(ok)ok=::renameat(root_,temporary,dir,file)==0;
        if(!ok){::unlinkat(root_,temporary,0);++write_errors;}else ++writes;
        ::close(dir);
    }
};

class StartSignatureDiskMemo:public StartSignatureMemo {
    std::string world_;
    bool eligible_=false;
    int maximum_=0;
    std::chrono::steady_clock::time_point started_;
    std::string full_key(const std::string& key)const{
        uint64_t n=world_.size();std::string full;
        const uint16_t endian=1;full.push_back(*reinterpret_cast<const char*>(&endian)?'L':'B');
        full.append(reinterpret_cast<const char*>(&n),sizeof(n));full+=world_;full+=key;return full;
    }
public:
    QueryDiskMemo disk;
    double minimum_seconds=.01;
    void prepare(double res,double inner,double outer,double sep,double margin,const double* bounds,int nw,const double* walls){
        StartSignatureMemo::prepare(res,inner,outer,sep,margin,bounds,nw,walls);
        const double fixed[]={res,inner,outer,sep,margin,bounds[0],bounds[1],bounds[2],bounds[3]};
        world_.assign(reinterpret_cast<const char*>(fixed),sizeof(fixed));
        if(nw)world_.append(reinterpret_cast<const char*>(walls),size_t(nw)*4*sizeof(double));
    }
    std::string key(int sx,int sy,int max_time,const double* goal,int nc,const double* constraints,uint16_t signature){
        eligible_=nc>0;maximum_=max_time+1;
        std::string result=StartSignatureMemo::key(sx,sy,max_time,goal,nc,constraints,signature);
        // Strengthen the already-exact key across processes with all x86
        // floating-point control bits, including DAZ/FTZ and x87 precision.
#if defined(__SSE2__)
        uint32_t control=_mm_getcsr()&0xffc0u;result.append(reinterpret_cast<const char*>(&control),sizeof(control));
#endif
#if defined(__x86_64__) || defined(__i386__)
        uint16_t x87;asm volatile("fnstcw %0":"=m"(x87));result.append(reinterpret_cast<const char*>(&x87),sizeof(x87));
#endif
        return result;
    }
    bool get(const std::string& key,const double* start,double* output,int& count){
        started_=std::chrono::steady_clock::now();
        if(StartSignatureMemo::get(key,start,output,count))return true;
        std::vector<double> answer;
        if(!eligible_ || !disk.get(full_key(key),maximum_,answer))return false;
        count=int(answer.size()/2);StartSignatureMemo::put(key,answer.data(),count);
        if(count){std::memcpy(output,answer.data(),answer.size()*sizeof(double));output[0]=start[0];output[1]=start[1];}else ++negative_hits;
        ++hits;return true;
    }
    void put(const std::string& key,const double* output,int count){
        StartSignatureMemo::put(key,output,count);
        double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started_).count();
        // Admission changes computation only. Cheap queries stay in memory.
        if(eligible_ && elapsed>=minimum_seconds)disk.put(full_key(key),output,count);
    }
};
static StartSignatureDiskMemo start_signature_disk_memo;
