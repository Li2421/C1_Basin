#include <cstdint>
#include <cfenv>
#include <xmmintrin.h>
extern "C" void bootstrap_fp_control(uint32_t* out){
 out[0]=uint32_t(std::fegetround());out[1]=_mm_getcsr()&0xffc0;
 unsigned short control;asm volatile("fnstcw %0":"=m"(control));out[2]=control;
}
