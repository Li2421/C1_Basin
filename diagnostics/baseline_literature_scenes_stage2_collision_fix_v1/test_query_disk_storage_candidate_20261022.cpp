// Tests only a new scratch namespace. No plant/controller calls or evidence edits.
#include "cbs_kernel_query_disk_candidate_20261022.cpp"
#include <cassert>
#include <sys/wait.h>
#include <fstream>
#include <sched.h>

static std::pair<std::string,std::string> collision(){
    std::unordered_map<uint16_t,std::string> seen;
    for(int n=0;n<65537;++n){
        std::string key="storage-test-key-"+std::to_string(n);unsigned char h[32];
        SHA256(reinterpret_cast<const unsigned char*>(key.data()),key.size(),h);
        uint16_t slot=uint16_t(h[0])*256+h[1];auto it=seen.find(slot);
        if(it!=seen.end())return {it->second,key};seen.emplace(slot,key);
    }
    std::abort();
}

int main(){
    cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(23,&cpus);assert(::sched_setaffinity(0,sizeof(cpus),&cpus)==0);
    const std::string ns(64,'c');
    static_assert(sizeof(double)==8);
    assert(ns.size()==64);QueryDiskMemo memo;assert(memo.configure(ns.c_str()));
    assert(!memo.configure("../../outside"));
    const double a[]={.1,.2,101.,202.},b[]={.3,.4,303.,404.};std::vector<double> out;
    auto keys=collision();memo.put(keys.first,a,2);assert(memo.get(keys.first,2,out));assert(out.size()==4 && out[0]==0 && out[1]==0 && out[2]==101. && out[3]==202.);
    memo.put(keys.second,b,2);assert(!memo.get(keys.first,2,out));assert(memo.get(keys.second,2,out));assert(out[2]==303. && out[3]==404.);
    memo.put("negative-result",nullptr,0);assert(memo.get("negative-result",2,out) && out.empty());
    assert(!memo.get(keys.second,1,out));
    memo.enable(false);assert(!memo.get(keys.second,2,out));memo.enable(true);
    const uint64_t skipped=memo.skipped;memo.put("oversize",a,2000);assert(memo.skipped==skipped+1);
    // Four processes race on two different exact keys in the same direct slot.
    for(int child=0;child<4;++child){
        pid_t pid=fork();assert(pid>=0);
        if(pid==0){
            QueryDiskMemo local;assert(local.configure(ns.c_str()));
            bool first=(child%2)==0;const auto& key=first?keys.first:keys.second;const double* value=first?a:b;
            for(int n=0;n<128;++n){local.put(key,value,2);if(local.get(key,2,out))assert(out.size()==4 && out[0]==0 && out[1]==0 && out[2]==value[2] && out[3]==value[3]);}
            _exit(0);
        }
    }
    for(int child=0;child<4;++child){int status=0;assert(wait(&status)>0 && WIFEXITED(status) && WEXITSTATUS(status)==0);}
    // Deliberate corruption is confined to this test-only scratch entry.
    memo.put(keys.first,a,2);unsigned char hash[32];SHA256(reinterpret_cast<const unsigned char*>(keys.first.data()),keys.first.size(),hash);
    char suffix[8];std::snprintf(suffix,sizeof(suffix),"%02x/%02x",unsigned(hash[0]),unsigned(hash[1]));
    std::string path="/home/zhihan/research/Basin_C1/diagnostics/baseline_literature_scenes_distributed_v2/runtime_cbs_query_memo_candidate_20261022/"+ns+"/"+suffix;
    int fd=::open(path.c_str(),O_WRONLY|O_NOFOLLOW);assert(fd>=0);unsigned char bad=0;assert(::pwrite(fd,&bad,1,20)==1);::close(fd);
    assert(!memo.get(keys.first,2,out));memo.put(keys.first,a,2);assert(memo.get(keys.first,2,out));
    // A truncated publication is rejected too.
    fd=::open(path.c_str(),O_WRONLY|O_TRUNC|O_NOFOLLOW);assert(fd>=0);assert(::write(fd,"bad",3)==3);::close(fd);assert(!memo.get(keys.first,2,out));
    std::puts("PASS: exact key,positive/negative results,slot collision,512 concurrent replacements,corruption,truncation,admission bounds,disabled fallback");
}
