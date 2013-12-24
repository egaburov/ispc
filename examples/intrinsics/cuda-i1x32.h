/*
  Copyright (c) 2010-2012, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
*/

#define FORCEINLINE __device__ __forceinline__

#define int64_t long long
#define uint64_t unsigned long long
typedef bool bool_t;
typedef bool_t __vec1_i1;
typedef float __vec1_f;
typedef double __vec1_d;
typedef int8_t __vec1_i8;
typedef int16_t __vec1_i16;
typedef int32_t __vec1_i32;
typedef int64_t __vec1_i64;

#define WARPSZ2 5
#define WARPSZ (1<<WARPSZ2)

static FORCEINLINE int32_t programIndex()
{
  int laneIdx;
  asm("mov.u32 %0, %laneid;" : "=r" (laneIdx));
  return laneIdx;
}
static FORCEINLINE int32_t __blockIndex0()
{
  return (blockIdx.x << 2) + (threadIdx.x >> WARPSZ2);
}
static FORCEINLINE int32_t __blockIndex1()
{
  return blockIdx.y;
}
static FORCEINLINE int32_t __blockIndex2()
{
  return blockIdx.z;
}
static FORCEINLINE int32_t __blockCount0()
{
  return gridDim.x << 2;
}
static FORCEINLINE int32_t __blockCount1()
{
  return gridDim.y;
}
static FORCEINLINE int32_t __blockCount2()
{
  return gridDim.z;
}
static FORCEINLINE int32_t __blockIndex()
{
  return __blockIndex0() + __blockCount0()*(__blockIndex1() + __blockCount1()*__blockIndex2());
}
static FORCEINLINE int32_t __blockCount()
{
  return __blockCount0()*__blockCount1()*__blockCount2();
}

static FORCEINLINE uint8_t* ISPCAlloc(uint8_t **handle, uint64_t align, uint64_t size)
{
  uint8_t *ptr = NULL;
  if (programIndex() == 0)
    ptr = static_cast<uint8_t*>(cudaGetParameterBuffer(align, size));
  return ptr;
}
static FORCEINLINE void ISPCLaunch(uint8_t **handle, uint8_t *func, uint8_t *params, uint32_t ntx, uint32_t nty, uint32_t ntz)
{
  const int nbx = (ntx+4-1)/4;
  const int nby =  nty;
  const int nbz =  ntz;
  const int sharedMemSize = 0;
  const cudaStream_t stream(0);

  if (programIndex() == 0)
      cudaLaunchDevice(func, params, dim3(nbx,nby,nbz), dim3(128,1,1), sharedMemSize, stream);
}
static FORCEINLINE void ISPCSync(uint8_t *handle)
{
  cudaDeviceSynchronize();
}

/****** shuffle instruction ******/

/* bool, int8_t, int16_t, int32_t */
template<typename T>
static FORCEINLINE T __shuffle(const T v, const int32_t index)
{
  return static_cast<T>(__shfl(static_cast<int32_t>(v), index));
}
/* float */
template<>
static FORCEINLINE float __shuffle(const float v, const int32_t index)
{
  return __shfl(v, index);
}
/* double */
template<>
static FORCEINLINE double __shuffle(const double v, const int32_t index)
{
  return __hiloint2double(
      __shfl(__double2hiint(v), index),
      __shfl(__double2loint(v), index));
}

/* int64_t */
  template<>
static FORCEINLINE int64_t __shuffle(const int64_t _v, const int32_t index)
{
  const double v = __longlong_as_double(_v);
  return __double_as_longlong(__hiloint2double(
      __shfl(__double2hiint(v), index),
      __shfl(__double2loint(v), index)));
}


/**** shared storage ****/

template<typename T>
struct SharedArray
{
  private:
    T *ptr;
  public:
    FORCEINLINE SharedArray(const size_t n)
    {
      if (programIndex() == 0)
        ptr = (T*)malloc(n * sizeof(T));
      uint64_t ptr_i64 = (uint64_t)ptr;
      ptr_i64 = __shuffle(ptr_i64,0);
      ptr = (T*)ptr_i64;
    }
    FORCEINLINE ~SharedArray()
    {
      if (programIndex() == 0)
        free(ptr);
    }
    const FORCEINLINE T& operator[](const size_t i) const { return ptr[i]; }
    FORCEINLINE       T& operator[](const size_t i) { return ptr[i]; }
    FORCEINLINE operator T*() { return ptr; }
    FORCEINLINE operator const T*() const { return ptr; }
};

/***** reductions *****/

template<typename T>
struct OpAdd
{
  static FORCEINLINE T exec(const T a, const T b) { return a + b; }
};
template<typename T>
struct OpMin
{
  static FORCEINLINE T exec(const T a, const T b) { return min(a,b); }
};
template<typename T>
struct OpMax
{
  static FORCEINLINE T exec(const T a, const T b) { return max(a,b); }
};

  template<typename T, typename OP>
static FORCEINLINE T __reduce(T v)
{
#pragma unroll 
  for (int i = WARPSZ-1; i >= 0; i++)
    v = OP::exec(v, __shuffle(v, 1<<i));
  return __shuffle(v, 0);
}
 
#if 0
template<typename T>
static FORCEINLINE T __set(const int index,
      const T v0,  const T v1,  const T v2,  const T v3,
      const T v4,  const T v5,  const T v6,  const T v7,
      const T v8,  const T v9,  const T v10, const T v11,
      const T v12, const T v13, const T v14, const T v15,
      const T v16, const T v17, const T v18, const T v19,
      const T v20, const T v21, const T v22, const T v23,
      const T v24, const T v25, const T v26, const T v27,
      const T v28, const T v29, const T v30, const T v31);
template<typename T>
static FORCEINLINE T __set(const int32_t index, const T value);
template<typename T>
static FORCEINLINE T __get(const int32_t index, const T value);
#endif

#if 0
template<>
static FORCEINLINE bool __set<bool>(const int index,
      const bool v0,  const bool v1,  const bool v2,  const bool v3,
      const bool v4,  const bool v5,  const bool v6,  const bool v7,
      const bool v8,  const bool v9,  const bool v10, const bool v11,
      const bool v12, const bool v13, const bool v14, const bool v15,
      const bool v16, const bool v17, const bool v18, const bool v19,
      const bool v20, const bool v21, const bool v22, const bool v23,
      const bool v24, const bool v25, const bool v26, const bool v27,
      const bool v28, const bool v29, const bool v30, const bool v31)
{
  const uint32_t v = ((v0 & 1) |
      ((v1 & 1) << 1) |
      ((v2 & 1) << 2) |
      ((v3 & 1) << 3) |
      ((v4 & 1) << 4) |
      ((v5 & 1) << 5) |
      ((v6 & 1) << 6) |
      ((v7 & 1) << 7) |
      ((v8 & 1) << 8) |
      ((v9 & 1) << 9) |
      ((v10 & 1) << 10) |
      ((v11 & 1) << 11) |
      ((v12 & 1) << 12) |
      ((v13 & 1) << 13) |
      ((v14 & 1) << 14) |
      ((v15 & 1) << 15) |
      ((v16 & 1) << 16) |
      ((v17 & 1) << 17) |
      ((v18 & 1) << 18) |
      ((v19 & 1) << 19) |
      ((v20 & 1) << 20) |
      ((v21 & 1) << 21) |
      ((v22 & 1) << 22) |
      ((v23 & 1) << 23) |
      ((v24 & 1) << 24) |
      ((v25 & 1) << 25) |
      ((v26 & 1) << 26) |
      ((v27 & 1) << 27) |
      ((v28 & 1) << 28) |
      ((v29 & 1) << 29) |
      ((v30 & 1) << 30) |
      ((v31 & 1) << 31));
  return v & (1<<index);
}
#endif

template <typename T>
struct vec32 
{
  FORCEINLINE vec32() { }
  FORCEINLINE vec32(T _v) : v(_v) {}
#if 0
  FORCEINLINE vec32(T v0,  T v1,  T v2,  T v3,  T v4,  T v5,  T v6,  T v7,
      T v8,  T v9,  T v10, T v11, T v12, T v13, T v14, T v15,
      T v16, T v17, T v18, T v19, T v20, T v21, T v22, T v23,
      T v24, T v25, T v26, T v27, T v28, T v29, T v30, T v31) :
    v(set<T>(
          v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,
          v8,  v9,  v10, v11, v12, v13, v14, v15,
          v16, v17, v18, v19, v20, v21, v22, v23,
          v24, v25, v26, v27, v28, v29, v30, v31)) { } 
#endif
  T v;
  FORCEINLINE T get(const int index)        const { return __shuffle<T>(v, index); }
#if 0
  FORCEINLINE void set(const int i, const T vv)  { v = vv; } // = __set<T>(i,vv); 
#endif
};

struct __vec32_i1 : public vec32<bool_t>
{
  FORCEINLINE __vec32_i1() { }
  FORCEINLINE __vec32_i1(bool_t v) : vec32<bool_t>(v) {}
};

struct __vec32_f : public vec32<float> 
{
  FORCEINLINE __vec32_f() { }
  FORCEINLINE __vec32_f(float v) : vec32<float>(v) {}
};

struct __vec32_d : public vec32<double> 
{
  FORCEINLINE __vec32_d() { }
  FORCEINLINE __vec32_d(double v) : vec32<double>(v) {}
};

struct __vec32_i8 : public vec32<int8_t> 
{
  FORCEINLINE __vec32_i8() { }
  FORCEINLINE __vec32_i8(int8_t v) : vec32<int8_t>(v) {}
};

struct __vec32_i16 : public vec32<int16_t>
{
  FORCEINLINE __vec32_i16() { }
  FORCEINLINE __vec32_i16(int16_t v) : vec32<int16_t>(v) {}
};

struct __vec32_i32 : public vec32<int32_t>
{
  FORCEINLINE __vec32_i32() { }
  FORCEINLINE __vec32_i32(int32_t v) : vec32<int32_t>(v) {}
};

struct __vec32_i64 : public vec32<int64_t>
{
  FORCEINLINE __vec32_i64() { }
  FORCEINLINE __vec32_i64(int64_t v) : vec32<int64_t>(v) {}
};

// static inline int32_t __extract_element(__vec32_i32, int);

///////////////////////////////////////////////////////////////////////////
// macros...

#define UNARY_OP(TYPE, NAME, OP)            \
static FORCEINLINE TYPE NAME(TYPE v) {      \
    TYPE ret;                               \
    ret.v = OP(v.v);                        \
    return ret;                             \
}

#define BINARY_OP(TYPE, NAME, OP)                               \
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                  \
    TYPE ret;                                                   \
    ret.v = a.v OP b.v;                                         \
    return ret;                                                 \
}

#define BINARY_OP_CAST(TYPE, CAST, NAME, OP)                        \
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                      \
   TYPE ret;                                                        \
   ret.v = (CAST)(a.v) OP (CAST)(b.v);                              \
   return ret;                                                      \
}

#define BINARY_OP_FUNC(TYPE, NAME, FUNC)                            \
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                      \
   TYPE ret;                                                        \
   ret.v = FUNC(a.v, b.v);                                          \
   return ret;                                                      \
}

#define CMP_OP(TYPE, SUFFIX, CAST, NAME, OP)                        \
static FORCEINLINE __vec32_i1 NAME##_##SUFFIX(TYPE a, TYPE b) {     \
   __vec32_i1 ret;                                                  \
   ret.v = ((CAST)(a.v) OP (CAST)(b.v));                            \
   return ret;                                                      \
}                                                                   \
static FORCEINLINE __vec32_i1 NAME##_##SUFFIX##_and_mask(TYPE a, TYPE b,       \
                                              __vec32_i1 mask) {    \
   __vec32_i1 ret;                                                  \
   ret.v = ((CAST)(a.v) OP (CAST)(b.v));                            \
   ret.v &= mask.v;                                                 \
   return ret;                                                      \
}

#if 0
#define INSERT_EXTRACT(VTYPE, STYPE)                                  \
static FORCEINLINE STYPE __extract_element(VTYPE v, int index) {      \
    return __shuffle<STYPE>(v.v,index)  ;                              \
}                                                                     \
static FORCEINLINE void __insert_element(VTYPE &v, int index, STYPE val) { \
    v.set(index, val);                                                \
}
#else
#define INSERT_EXTRACT(VTYPE, STYPE)                                  \
static FORCEINLINE STYPE __extract_element(VTYPE v, int index) {      \
    return __shuffle<STYPE>(v.v,index)  ;                              \
}                                                                     
#endif

#define LOAD_STORE(VTYPE, STYPE)                       \
template <int ALIGN>                                   \
static FORCEINLINE VTYPE __load(const VTYPE *p) {      \
    STYPE *ptr = (STYPE *)p;                           \
    VTYPE ret;                                         \
    ret.v = ptr[programIndex()];                       \
    return ret;                                        \
}                                                      \
template <int ALIGN>                                   \
static FORCEINLINE void __store(VTYPE *p, VTYPE v) {   \
    STYPE *ptr = (STYPE *)p;                           \
    ptr[programIndex()] = v.v;                         \
}

#define REDUCE_ADD(TYPE, VTYPE, NAME)           \
static FORCEINLINE TYPE NAME(VTYPE v) {         \
     TYPE ret = __reduce<TYPE,OpAdd<TYPE> >(v.v);        \
     return ret;                                \
}

#define REDUCE_MIN(TYPE, VTYPE, NAME, OP)                       \
static FORCEINLINE TYPE NAME(VTYPE v) {                         \
    TYPE ret = __reduce<TYPE,OpMin<TYPE> >(v.v);                         \
    return ret;                                                 \
}
#define REDUCE_MAX(TYPE, VTYPE, NAME, OP)                       \
static FORCEINLINE TYPE NAME(VTYPE v) {                         \
    TYPE ret = __reduce<TYPE,OpMax<TYPE> >(v.v);                         \
    return ret;                                                 \
}

#define SELECT(TYPE)                                                \
static FORCEINLINE TYPE __select(__vec32_i1 mask, TYPE a, TYPE b) { \
    TYPE ret;                                                       \
    ret.v = mask.v ? a.v : b.v;                                     \
    return ret;                                                     \
}                                                                   \
static FORCEINLINE TYPE __select(bool cond, TYPE a, TYPE b) {       \
    return cond ? a : b;                                            \
}

#define SHIFT_UNIFORM(TYPE, CAST, NAME, OP)                         \
static FORCEINLINE TYPE NAME(TYPE a, int32_t b) {                   \
   TYPE ret;                                                        \
   ret.v = (CAST)(a.v) OP b;                                        \
   return ret;                                                      \
}

#define SMEAR(VTYPE, NAME, STYPE)                                  \
template <class RetVecType> VTYPE __smear_##NAME(STYPE);           \
template <> FORCEINLINE VTYPE __smear_##NAME<VTYPE>(STYPE v) {     \
    VTYPE ret;                                                     \
    ret.v = v;                                                     \
    return ret;                                                    \
}

#define SETZERO(VTYPE, NAME)                                       \
template <class RetVecType> VTYPE __setzero_##NAME();              \
template <> FORCEINLINE VTYPE __setzero_##NAME<VTYPE>() {          \
    VTYPE ret;                                                     \
    ret.v = 0;                                                     \
    return ret;                                                    \
}

#define UNDEF(VTYPE, NAME)                                         \
template <class RetVecType> VTYPE __undef_##NAME();                \
template <> FORCEINLINE VTYPE __undef_##NAME<VTYPE>() {            \
    return VTYPE();                                                \
}

#define BROADCAST(VTYPE, NAME, STYPE)                 \
static FORCEINLINE VTYPE __broadcast_##NAME(VTYPE v, int index) {   \
    VTYPE ret;                                        \
    ret.v = __shuffle<STYPE>(v.v, index & 0x1F);             \
    return ret;                                       \
}                                                     \

#define ROTATE(VTYPE, NAME, STYPE)                    \
static FORCEINLINE VTYPE __rotate_##NAME(VTYPE v, int index) {   \
    VTYPE ret;                                        \
    ret.v = __shuffle<STYPE>(v.v, (programIndex() + index) & 0x1F); \
    return ret;                                       \
}                                                     \

#define SHIFT(VTYPE, NAME, STYPE)                     \
static FORCEINLINE VTYPE __shift_##NAME(VTYPE v, int index) {   \
    VTYPE ret;                                        \
    const int srcLane = programIndex() + index;       \
    ret.v = __shuffle<STYPE>(v.v, srcLane);           \
    if (srcLane < 0 || srcLane >= 32) ret.v = 0;      \
    return ret;                                       \
}                                                     \

#define SHUFFLES(VTYPE, NAME, STYPE)                  \
static FORCEINLINE VTYPE __shuffle_##NAME(VTYPE v, __vec32_i32 index) {   \
    VTYPE ret;                                        \
    ret.v = __shuffle<STYPE>(v.v, index.v & 0x1F);    \
    return ret;                                       \
}                                                     \
static FORCEINLINE VTYPE __shuffle2_##NAME(VTYPE v0, VTYPE v1, __vec32_i32 index) {     \
    VTYPE ret0, ret1, ret;                            \
    const int ii = index.v & 0x3F;                    \
    ret0.v = __shuffle<STYPE>(v0.v, ii   );           \
    ret1.v = __shuffle<STYPE>(v1.v, ii-31);           \
    ret. v = ii < 0x1F ? ret0.v : ret1.v;             \
    return ret;                                       \
}

///////////////////////////////////////////////////////////////////////////

#define INSERT_EXTRACT1(VTYPE, STYPE)                                  \
static FORCEINLINE STYPE __extract_element(VTYPE v, int index) {      \
    return __shuffle<STYPE>(v,index)  ;                                      \
}                                                                     \
static FORCEINLINE void __insert_element(VTYPE &v, int index, STYPE val) { \
    v = val;                                                \
}
INSERT_EXTRACT1(__vec1_i1, bool)
INSERT_EXTRACT1(__vec1_i8, int8_t)
INSERT_EXTRACT1(__vec1_i16, int16_t)
INSERT_EXTRACT1(__vec1_i32, int32_t)
INSERT_EXTRACT1(__vec1_i64, int64_t)
INSERT_EXTRACT1(__vec1_f, float)
INSERT_EXTRACT1(__vec1_d, double)

///////////////////////////////////////////////////////////////////////////
// mask ops

static FORCEINLINE uint64_t __movmsk(__vec32_i1 mask) { return  __ballot(mask.v); }
static FORCEINLINE bool __any(__vec32_i1 mask) {return  __any(mask.v); }
static FORCEINLINE bool __all(__vec32_i1 mask) { return __all(mask.v); }
static FORCEINLINE bool __none(__vec32_i1 mask) { return  !__any(mask.v); }
static FORCEINLINE __vec32_i1 __equal_i1(__vec32_i1 a, __vec32_i1 b) {  return static_cast<bool_t>(a.v == b.v); }
static FORCEINLINE __vec32_i1 __and(__vec32_i1 a, __vec32_i1 b) {  return static_cast<bool_t>(a.v & b.v); }
static FORCEINLINE __vec32_i1 __xor(__vec32_i1 a, __vec32_i1 b) {   return a.v ^ b.v; }
static FORCEINLINE __vec32_i1 __or(__vec32_i1 a, __vec32_i1 b) {   return a.v | b.v; }
static FORCEINLINE __vec32_i1 __not(__vec32_i1 v) { return ~v.v; }
static FORCEINLINE __vec32_i1 __and_not1(__vec32_i1 a, __vec32_i1 b) { return ~a.v & b.v; }
static FORCEINLINE __vec32_i1 __and_not2(__vec32_i1 a, __vec32_i1 b) { return  a.v & ~b.v; }
static FORCEINLINE __vec32_i1 __select(__vec32_i1 mask, __vec32_i1 a, 
                                       __vec32_i1 b) { return (a.v & mask.v) | (b.v & ~mask.v); }
static FORCEINLINE __vec32_i1 __select(bool cond, __vec32_i1 a, __vec32_i1 b) { return cond ? a : b; }

LOAD_STORE(__vec32_i1, bool);

template <class RetVecType> __vec32_i1 __smear_i1(int i);
template <> FORCEINLINE __vec32_i1 __smear_i1<__vec32_i1>(int v) {
    return __vec32_i1(v);
}

template <class RetVecType> __vec32_i1 __setzero_i1();
template <> FORCEINLINE __vec32_i1 __setzero_i1<__vec32_i1>() {
    return __vec32_i1(0);
}

template <class RetVecType> __vec32_i1 __undef_i1();
template <> FORCEINLINE __vec32_i1 __undef_i1<__vec32_i1>() {
    return __vec32_i1();
}


///////////////////////////////////////////////////////////////////////////
// int8

BINARY_OP(__vec32_i8, __add, +)
BINARY_OP(__vec32_i8, __sub, -)
BINARY_OP(__vec32_i8, __mul, *)

BINARY_OP(__vec32_i8, __or, |)
BINARY_OP(__vec32_i8, __and, &)
BINARY_OP(__vec32_i8, __xor, ^)
BINARY_OP(__vec32_i8, __shl, <<)

BINARY_OP_CAST(__vec32_i8, uint8_t, __udiv, /)
BINARY_OP_CAST(__vec32_i8, int8_t,  __sdiv, /)

BINARY_OP_CAST(__vec32_i8, uint8_t, __urem, %)
BINARY_OP_CAST(__vec32_i8, int8_t,  __srem, %)
BINARY_OP_CAST(__vec32_i8, uint8_t, __lshr, >>)
BINARY_OP_CAST(__vec32_i8, int8_t,  __ashr, >>)

SHIFT_UNIFORM(__vec32_i8, uint8_t, __lshr, >>)
SHIFT_UNIFORM(__vec32_i8, int8_t, __ashr, >>)
SHIFT_UNIFORM(__vec32_i8, int8_t, __shl, <<)

CMP_OP(__vec32_i8, i8, int8_t,  __equal, ==)
CMP_OP(__vec32_i8, i8, int8_t,  __not_equal, !=)
CMP_OP(__vec32_i8, i8, uint8_t, __unsigned_less_equal, <=)
CMP_OP(__vec32_i8, i8, int8_t,  __signed_less_equal, <=)
CMP_OP(__vec32_i8, i8, uint8_t, __unsigned_greater_equal, >=)
CMP_OP(__vec32_i8, i8, int8_t,  __signed_greater_equal, >=)
CMP_OP(__vec32_i8, i8, uint8_t, __unsigned_less_than, <)
CMP_OP(__vec32_i8, i8, int8_t,  __signed_less_than, <)
CMP_OP(__vec32_i8, i8, uint8_t, __unsigned_greater_than, >)
CMP_OP(__vec32_i8, i8, int8_t,  __signed_greater_than, >)

SELECT(__vec32_i8)
INSERT_EXTRACT(__vec32_i8, int8_t)
SMEAR(__vec32_i8, i8, int8_t)
SETZERO(__vec32_i8, i8)
UNDEF(__vec32_i8, i8)
BROADCAST(__vec32_i8, i8, int8_t)
ROTATE(__vec32_i8, i8, int8_t)
SHIFT(__vec32_i8, i8, int8_t)
SHUFFLES(__vec32_i8, i8, int8_t)
LOAD_STORE(__vec32_i8, int8_t)

///////////////////////////////////////////////////////////////////////////
// int16

BINARY_OP(__vec32_i16, __add, +)
BINARY_OP(__vec32_i16, __sub, -)
BINARY_OP(__vec32_i16, __mul, *)

BINARY_OP(__vec32_i16, __or, |)
BINARY_OP(__vec32_i16, __and, &)
BINARY_OP(__vec32_i16, __xor, ^)
BINARY_OP(__vec32_i16, __shl, <<)

BINARY_OP_CAST(__vec32_i16, uint16_t, __udiv, /)
BINARY_OP_CAST(__vec32_i16, int16_t,  __sdiv, /)

BINARY_OP_CAST(__vec32_i16, uint16_t, __urem, %)
BINARY_OP_CAST(__vec32_i16, int16_t,  __srem, %)
BINARY_OP_CAST(__vec32_i16, uint16_t, __lshr, >>)
BINARY_OP_CAST(__vec32_i16, int16_t,  __ashr, >>)

SHIFT_UNIFORM(__vec32_i16, uint16_t, __lshr, >>)
SHIFT_UNIFORM(__vec32_i16, int16_t, __ashr, >>)
SHIFT_UNIFORM(__vec32_i16, int16_t, __shl, <<)

CMP_OP(__vec32_i16, i16, int16_t,  __equal, ==)
CMP_OP(__vec32_i16, i16, int16_t,  __not_equal, !=)
CMP_OP(__vec32_i16, i16, uint16_t, __unsigned_less_equal, <=)
CMP_OP(__vec32_i16, i16, int16_t,  __signed_less_equal, <=)
CMP_OP(__vec32_i16, i16, uint16_t, __unsigned_greater_equal, >=)
CMP_OP(__vec32_i16, i16, int16_t,  __signed_greater_equal, >=)
CMP_OP(__vec32_i16, i16, uint16_t, __unsigned_less_than, <)
CMP_OP(__vec32_i16, i16, int16_t,  __signed_less_than, <)
CMP_OP(__vec32_i16, i16, uint16_t, __unsigned_greater_than, >)
CMP_OP(__vec32_i16, i16, int16_t,  __signed_greater_than, >)

SELECT(__vec32_i16)
INSERT_EXTRACT(__vec32_i16, int16_t)
SMEAR(__vec32_i16, i16, int16_t)
SETZERO(__vec32_i16, i16)
UNDEF(__vec32_i16, i16)
BROADCAST(__vec32_i16, i16, int16_t)
ROTATE(__vec32_i16, i16, int16_t)
SHIFT(__vec32_i16, i16, int16_t)
SHUFFLES(__vec32_i16, i16, int16_t)
LOAD_STORE(__vec32_i16, int16_t)

///////////////////////////////////////////////////////////////////////////
// int32

BINARY_OP(__vec32_i32, __add, +)
BINARY_OP(__vec32_i32, __sub, -)
BINARY_OP(__vec32_i32, __mul, *)

BINARY_OP(__vec32_i32, __or, |)
BINARY_OP(__vec32_i32, __and, &)
BINARY_OP(__vec32_i32, __xor, ^)
BINARY_OP(__vec32_i32, __shl, <<)

BINARY_OP_CAST(__vec32_i32, uint32_t, __udiv, /)
BINARY_OP_CAST(__vec32_i32, int32_t,  __sdiv, /)

BINARY_OP_CAST(__vec32_i32, uint32_t, __urem, %)
BINARY_OP_CAST(__vec32_i32, int32_t,  __srem, %)
BINARY_OP_CAST(__vec32_i32, uint32_t, __lshr, >>)
BINARY_OP_CAST(__vec32_i32, int32_t,  __ashr, >>)

SHIFT_UNIFORM(__vec32_i32, uint32_t, __lshr, >>)
SHIFT_UNIFORM(__vec32_i32, int32_t, __ashr, >>)
SHIFT_UNIFORM(__vec32_i32, int32_t, __shl, <<)

CMP_OP(__vec32_i32, i32, int32_t,  __equal, ==)
CMP_OP(__vec32_i32, i32, int32_t,  __not_equal, !=)
CMP_OP(__vec32_i32, i32, uint32_t, __unsigned_less_equal, <=)
CMP_OP(__vec32_i32, i32, int32_t,  __signed_less_equal, <=)
CMP_OP(__vec32_i32, i32, uint32_t, __unsigned_greater_equal, >=)
CMP_OP(__vec32_i32, i32, int32_t,  __signed_greater_equal, >=)
CMP_OP(__vec32_i32, i32, uint32_t, __unsigned_less_than, <)
CMP_OP(__vec32_i32, i32, int32_t,  __signed_less_than, <)
CMP_OP(__vec32_i32, i32, uint32_t, __unsigned_greater_than, >)
CMP_OP(__vec32_i32, i32, int32_t,  __signed_greater_than, >)

SELECT(__vec32_i32)
INSERT_EXTRACT(__vec32_i32, int32_t)
SMEAR(__vec32_i32, i32, int32_t)
SETZERO(__vec32_i32, i32)
UNDEF(__vec32_i32, i32)
BROADCAST(__vec32_i32, i32, int32_t)
ROTATE(__vec32_i32, i32, int32_t)
SHIFT(__vec32_i32, i32, int32_t)
SHUFFLES(__vec32_i32, i32, int32_t)
LOAD_STORE(__vec32_i32, int32_t)

///////////////////////////////////////////////////////////////////////////
// int64

BINARY_OP(__vec32_i64, __add, +)
BINARY_OP(__vec32_i64, __sub, -)
BINARY_OP(__vec32_i64, __mul, *)

BINARY_OP(__vec32_i64, __or, |)
BINARY_OP(__vec32_i64, __and, &)
BINARY_OP(__vec32_i64, __xor, ^)
BINARY_OP(__vec32_i64, __shl, <<)

BINARY_OP_CAST(__vec32_i64, uint64_t, __udiv, /)
BINARY_OP_CAST(__vec32_i64, int64_t,  __sdiv, /)

BINARY_OP_CAST(__vec32_i64, uint64_t, __urem, %)
BINARY_OP_CAST(__vec32_i64, int64_t,  __srem, %)
BINARY_OP_CAST(__vec32_i64, uint64_t, __lshr, >>)
BINARY_OP_CAST(__vec32_i64, int64_t,  __ashr, >>)

SHIFT_UNIFORM(__vec32_i64, uint64_t, __lshr, >>)
SHIFT_UNIFORM(__vec32_i64, int64_t, __ashr, >>)
SHIFT_UNIFORM(__vec32_i64, int64_t, __shl, <<)

CMP_OP(__vec32_i64, i64, int64_t,  __equal, ==)
CMP_OP(__vec32_i64, i64, int64_t,  __not_equal, !=)
CMP_OP(__vec32_i64, i64, uint64_t, __unsigned_less_equal, <=)
CMP_OP(__vec32_i64, i64, int64_t,  __signed_less_equal, <=)
CMP_OP(__vec32_i64, i64, uint64_t, __unsigned_greater_equal, >=)
CMP_OP(__vec32_i64, i64, int64_t,  __signed_greater_equal, >=)
CMP_OP(__vec32_i64, i64, uint64_t, __unsigned_less_than, <)
CMP_OP(__vec32_i64, i64, int64_t,  __signed_less_than, <)
CMP_OP(__vec32_i64, i64, uint64_t, __unsigned_greater_than, >)
CMP_OP(__vec32_i64, i64, int64_t,  __signed_greater_than, >)

SELECT(__vec32_i64)
INSERT_EXTRACT(__vec32_i64, int64_t)
SMEAR(__vec32_i64, i64, int64_t)
SETZERO(__vec32_i64, i64)
UNDEF(__vec32_i64, i64)
BROADCAST(__vec32_i64, i64, int64_t)
ROTATE(__vec32_i64, i64, int64_t)
SHIFT(__vec32_i64, i64, int64_t)
SHUFFLES(__vec32_i64, i64, int64_t)
LOAD_STORE(__vec32_i64, int64_t)

///////////////////////////////////////////////////////////////////////////
// float

BINARY_OP(__vec32_f, __add, +)
BINARY_OP(__vec32_f, __sub, -)
BINARY_OP(__vec32_f, __mul, *)
BINARY_OP(__vec32_f, __div, /)

CMP_OP(__vec32_f, float, float, __equal, ==)
CMP_OP(__vec32_f, float, float, __not_equal, !=)
CMP_OP(__vec32_f, float, float, __less_than, <)
CMP_OP(__vec32_f, float, float, __less_equal, <=)
CMP_OP(__vec32_f, float, float, __greater_than, >)
CMP_OP(__vec32_f, float, float, __greater_equal, >=)

static FORCEINLINE __vec32_i1 __ordered_float(__vec32_f a, __vec32_f b) {
    __vec32_i1 ret;
    ret.v = (a.v == a.v) && (b.v == b.v);
    return ret;
}

static FORCEINLINE __vec32_i1 __unordered_float(__vec32_f a, __vec32_f b) {
  __vec32_i1 ret;
  ret.v = (a.v != a.v) || (b.v != b.v);
  return ret;
}

SELECT(__vec32_f)
INSERT_EXTRACT(__vec32_f, float)
SMEAR(__vec32_f, float, float)
SETZERO(__vec32_f, float)
UNDEF(__vec32_f, float)
BROADCAST(__vec32_f, float, float)
ROTATE(__vec32_f, float, float)
SHIFT(__vec32_f, float, float)
SHUFFLES(__vec32_f, float, float)
LOAD_STORE(__vec32_f, float)

static FORCEINLINE float __exp_uniform_float(float v) {
#ifdef ISPC_FAST_MATH
  return __expf(v);
#else
  return expf(v);
#endif
}
static FORCEINLINE __vec32_f __exp_varying_float(__vec32_f v) {
  __vec32_f ret;
  ret.v = __exp_uniform_float(v.v);
  return ret;
}

static FORCEINLINE float __log_uniform_float(float v) {
#ifdef ISPC_FAST_MATH
  return __logf(v);
#else
  return logf(v);
#endif
}
static FORCEINLINE __vec32_f __log_varying_float(__vec32_f v) {
  __vec32_f ret;
  ret.v = __log_uniform_float(v.v);
  return ret;
}

static FORCEINLINE float __pow_uniform_float(float a, float b) {
#ifdef ISPC_FAST_MATH
  return powf(a, b);
#else
  return __powf(a, b);
#endif
}
static FORCEINLINE __vec32_f __pow_varying_float(__vec32_f a, __vec32_f b) {
  __vec32_f ret;
  ret.v = __pow_uniform_float(a.v, b.v);
  return ret;
}

static FORCEINLINE int __intbits(float v) 
{
  return __float_as_int(v);
}
static FORCEINLINE float __floatbits(int v) {
  return __int_as_float(v);
}

static FORCEINLINE float __half_to_float_uniform(int16_t h) 
{
  return __half2float(h);
}
static FORCEINLINE __vec32_f __half_to_float_varying(__vec32_i16 v) {
  __vec32_f ret;
  ret.v = __half_to_float_uniform(v.v);
  return ret;
}

static FORCEINLINE int16_t __float_to_half_uniform(float f) {
  return __float2half_rn(f);
}
static FORCEINLINE __vec32_i16 __float_to_half_varying(__vec32_f v) {
  __vec32_i16 ret;
  ret.v = __float_to_half_uniform(v.v);
  return ret;
}


///////////////////////////////////////////////////////////////////////////
// double

BINARY_OP(__vec32_d, __add, +)
BINARY_OP(__vec32_d, __sub, -)
BINARY_OP(__vec32_d, __mul, *)
BINARY_OP(__vec32_d, __div, /)

CMP_OP(__vec32_d, double, double, __equal, ==)
CMP_OP(__vec32_d, double, double, __not_equal, !=)
CMP_OP(__vec32_d, double, double, __less_than, <)
CMP_OP(__vec32_d, double, double, __less_equal, <=)
CMP_OP(__vec32_d, double, double, __greater_than, >)
CMP_OP(__vec32_d, double, double, __greater_equal, >=)

static FORCEINLINE __vec32_i1 __ordered_double(__vec32_d a, __vec32_d b) {
  __vec32_i1 ret;
  ret.v = (a.v == a.v) && (b.v == b.v);
  return ret;
}

static FORCEINLINE __vec32_i1 __unordered_double(__vec32_d a, __vec32_d b) {
  __vec32_i1 ret;
  ret.v = (a.v != a.v) || (b.v != b.v);
  return ret;
}
SELECT(__vec32_d)
INSERT_EXTRACT(__vec32_d, double)
SMEAR(__vec32_d, double, double)
SETZERO(__vec32_d, double)
UNDEF(__vec32_d, double)
BROADCAST(__vec32_d, double, double)
ROTATE(__vec32_d, double, double)
SHIFT(__vec32_d, double, double)
SHUFFLES(__vec32_d, double, double)
LOAD_STORE(__vec32_d, double)

///////////////////////////////////////////////////////////////////////////
// casts


#define CAST(TO, STO, FROM, SFROM, FUNC)        \
static FORCEINLINE TO FUNC(TO, FROM val) {      \
    TO ret;                                     \
    ret.v = (STO)((SFROM)(val.v));              \
    return ret;                                 \
}

// sign extension conversions
CAST(__vec32_i64, int64_t, __vec32_i32, int32_t, __cast_sext)
CAST(__vec32_i64, int64_t, __vec32_i16, int16_t, __cast_sext)
CAST(__vec32_i64, int64_t, __vec32_i8,  int8_t,  __cast_sext)
CAST(__vec32_i32, int32_t, __vec32_i16, int16_t, __cast_sext)
CAST(__vec32_i32, int32_t, __vec32_i8,  int8_t,  __cast_sext)
CAST(__vec32_i16, int16_t, __vec32_i8,  int8_t,  __cast_sext)

#define CAST_SEXT_I1(TYPE)                            \
static FORCEINLINE TYPE __cast_sext(TYPE, __vec32_i1 v) {  \
    TYPE ret;                                         \
    ret.v = 0;                                        \
    if (v.v)                                          \
       ret.v = ~ret.v;                                \
    return ret;                                       \
}

CAST_SEXT_I1(__vec32_i8)
CAST_SEXT_I1(__vec32_i16)
CAST_SEXT_I1(__vec32_i32)
CAST_SEXT_I1(__vec32_i64)

// zero extension
CAST(__vec32_i64, uint64_t, __vec32_i32, uint32_t, __cast_zext)
CAST(__vec32_i64, uint64_t, __vec32_i16, uint16_t, __cast_zext)
CAST(__vec32_i64, uint64_t, __vec32_i8,  uint8_t,  __cast_zext)
CAST(__vec32_i32, uint32_t, __vec32_i16, uint16_t, __cast_zext)
CAST(__vec32_i32, uint32_t, __vec32_i8,  uint8_t,  __cast_zext)
CAST(__vec32_i16, uint16_t, __vec32_i8,  uint8_t,  __cast_zext)

#define CAST_ZEXT_I1(TYPE)                            \
static FORCEINLINE TYPE __cast_zext(TYPE, __vec32_i1 v) {  \
    TYPE ret;                                         \
    ret.v = (v.v) ? 1 : 0;                            \
    return ret;                                       \
}

CAST_ZEXT_I1(__vec32_i8)
CAST_ZEXT_I1(__vec32_i16)
CAST_ZEXT_I1(__vec32_i32)
CAST_ZEXT_I1(__vec32_i64)

// truncations
CAST(__vec32_i32, int32_t, __vec32_i64, int64_t, __cast_trunc)
CAST(__vec32_i16, int16_t, __vec32_i64, int64_t, __cast_trunc)
CAST(__vec32_i8,  int8_t,  __vec32_i64, int64_t, __cast_trunc)
CAST(__vec32_i16, int16_t, __vec32_i32, int32_t, __cast_trunc)
CAST(__vec32_i8,  int8_t,  __vec32_i32, int32_t, __cast_trunc)
CAST(__vec32_i8,  int8_t,  __vec32_i16, int16_t, __cast_trunc)

// signed int to float/double
CAST(__vec32_f, float, __vec32_i8,   int8_t,  __cast_sitofp)
CAST(__vec32_f, float, __vec32_i16,  int16_t, __cast_sitofp)
CAST(__vec32_f, float, __vec32_i32,  int32_t, __cast_sitofp)
CAST(__vec32_f, float, __vec32_i64,  int64_t, __cast_sitofp)
CAST(__vec32_d, double, __vec32_i8,  int8_t,  __cast_sitofp)
CAST(__vec32_d, double, __vec32_i16, int16_t, __cast_sitofp)
CAST(__vec32_d, double, __vec32_i32, int32_t, __cast_sitofp)
CAST(__vec32_d, double, __vec32_i64, int64_t, __cast_sitofp)

// unsigned int to float/double
CAST(__vec32_f, float, __vec32_i8,   uint8_t,  __cast_uitofp)
CAST(__vec32_f, float, __vec32_i16,  uint16_t, __cast_uitofp)
CAST(__vec32_f, float, __vec32_i32,  uint32_t, __cast_uitofp)
CAST(__vec32_f, float, __vec32_i64,  uint64_t, __cast_uitofp)
CAST(__vec32_d, double, __vec32_i8,  uint8_t,  __cast_uitofp)
CAST(__vec32_d, double, __vec32_i16, uint16_t, __cast_uitofp)
CAST(__vec32_d, double, __vec32_i32, uint32_t, __cast_uitofp)
CAST(__vec32_d, double, __vec32_i64, uint64_t, __cast_uitofp)

static FORCEINLINE __vec32_f __cast_uitofp(__vec32_f, __vec32_i1 v) {
  __vec32_f ret;
  ret.v = (v.v) ? 1.0f : 0.0f;
  return ret;
}

// float/double to signed int
CAST(__vec32_i8,  int8_t,  __vec32_f, float, __cast_fptosi)
CAST(__vec32_i16, int16_t, __vec32_f, float, __cast_fptosi)
CAST(__vec32_i32, int32_t, __vec32_f, float, __cast_fptosi)
CAST(__vec32_i64, int64_t, __vec32_f, float, __cast_fptosi)
CAST(__vec32_i8,  int8_t,  __vec32_d, double, __cast_fptosi)
CAST(__vec32_i16, int16_t, __vec32_d, double, __cast_fptosi)
CAST(__vec32_i32, int32_t, __vec32_d, double, __cast_fptosi)
CAST(__vec32_i64, int64_t, __vec32_d, double, __cast_fptosi)

// float/double to unsigned int
CAST(__vec32_i8,  uint8_t,  __vec32_f, float, __cast_fptoui)
CAST(__vec32_i16, uint16_t, __vec32_f, float, __cast_fptoui)
CAST(__vec32_i32, uint32_t, __vec32_f, float, __cast_fptoui)
CAST(__vec32_i64, uint64_t, __vec32_f, float, __cast_fptoui)
CAST(__vec32_i8,  uint8_t,  __vec32_d, double, __cast_fptoui)
CAST(__vec32_i16, uint16_t, __vec32_d, double, __cast_fptoui)
CAST(__vec32_i32, uint32_t, __vec32_d, double, __cast_fptoui)
CAST(__vec32_i64, uint64_t, __vec32_d, double, __cast_fptoui)

// float/double conversions
CAST(__vec32_f, float,  __vec32_d, double, __cast_fptrunc)
CAST(__vec32_d, double, __vec32_f, float,  __cast_fpext)

typedef union {
    int32_t i32;
    float f;
    int64_t i64;
    double d;
} BitcastUnion;

#define CAST_BITS(TO, TO_ELT, FROM, FROM_ELT)       \
static FORCEINLINE TO __cast_bits(TO, FROM val) {   \
    TO r;                                           \
    BitcastUnion u;                                 \
    u.FROM_ELT = val.v;                             \
    r.v = u.TO_ELT;                                 \
    return r;                                       \
}

CAST_BITS(__vec32_f,   f,   __vec32_i32, i32)
CAST_BITS(__vec32_i32, i32, __vec32_f,   f)
CAST_BITS(__vec32_d,   d,   __vec32_i64, i64)
CAST_BITS(__vec32_i64, i64, __vec32_d,   d)

#define CAST_BITS_SCALAR(TO, FROM)                  \
static FORCEINLINE TO __cast_bits(TO, FROM v) {     \
    union {                                         \
    TO to;                                          \
    FROM from;                                      \
    } u;                                            \
    u.from = v;                                     \
    return u.to;                                    \
}

CAST_BITS_SCALAR(uint32_t, float)
CAST_BITS_SCALAR(int32_t, float)
CAST_BITS_SCALAR(float, uint32_t)
CAST_BITS_SCALAR(float, int32_t)
CAST_BITS_SCALAR(uint64_t, double)
CAST_BITS_SCALAR(int64_t, double)
CAST_BITS_SCALAR(double, uint64_t)
CAST_BITS_SCALAR(double, int64_t)

///////////////////////////////////////////////////////////////////////////
// various math functions

static FORCEINLINE void __fastmath() {
}

static FORCEINLINE float __round_uniform_float(float v) {
  return roundf(v);
}

static FORCEINLINE float __floor_uniform_float(float v)  {
  return floorf(v);
}

static FORCEINLINE float __ceil_uniform_float(float v) {
  return ceilf(v);
}

static FORCEINLINE double __round_uniform_double(double v) {
  return round(v);
}

static FORCEINLINE double __floor_uniform_double(double v) {
  return floor(v);
}

static FORCEINLINE double __ceil_uniform_double(double v) {
  return ceil(v);
}

UNARY_OP(__vec32_f, __round_varying_float, roundf)
UNARY_OP(__vec32_f, __floor_varying_float, floorf)
UNARY_OP(__vec32_f, __ceil_varying_float, ceilf)
UNARY_OP(__vec32_d, __round_varying_double, round)
UNARY_OP(__vec32_d, __floor_varying_double, floor)
UNARY_OP(__vec32_d, __ceil_varying_double, ceil)

// min/max

static FORCEINLINE float __min_uniform_float(float a, float b) { return (a<b) ? a : b; }
static FORCEINLINE float __max_uniform_float(float a, float b) { return (a>b) ? a : b; }
static FORCEINLINE double __min_uniform_double(double a, double b) { return (a<b) ? a : b; }
static FORCEINLINE double __max_uniform_double(double a, double b) { return (a>b) ? a : b; }

static FORCEINLINE int32_t __min_uniform_int32(int32_t a, int32_t b) { return (a<b) ? a : b; }
static FORCEINLINE int32_t __max_uniform_int32(int32_t a, int32_t b) { return (a>b) ? a : b; }
static FORCEINLINE int32_t __min_uniform_uint32(uint32_t a, uint32_t b) { return (a<b) ? a : b; }
static FORCEINLINE int32_t __max_uniform_uint32(uint32_t a, uint32_t b) { return (a>b) ? a : b; }

static FORCEINLINE int64_t __min_uniform_int64(int64_t a, int64_t b) { return (a<b) ? a : b; }
static FORCEINLINE int64_t __max_uniform_int64(int64_t a, int64_t b) { return (a>b) ? a : b; }
static FORCEINLINE int64_t __min_uniform_uint64(uint64_t a, uint64_t b) { return (a<b) ? a : b; }
static FORCEINLINE int64_t __max_uniform_uint64(uint64_t a, uint64_t b) { return (a>b) ? a : b; }


BINARY_OP_FUNC(__vec32_f, __max_varying_float, __max_uniform_float)
BINARY_OP_FUNC(__vec32_f, __min_varying_float, __min_uniform_float)
BINARY_OP_FUNC(__vec32_d, __max_varying_double, __max_uniform_double)
BINARY_OP_FUNC(__vec32_d, __min_varying_double, __min_uniform_double)

BINARY_OP_FUNC(__vec32_i32, __max_varying_int32, __max_uniform_int32)
BINARY_OP_FUNC(__vec32_i32, __min_varying_int32, __min_uniform_int32)
BINARY_OP_FUNC(__vec32_i32, __max_varying_uint32, __max_uniform_uint32)
BINARY_OP_FUNC(__vec32_i32, __min_varying_uint32, __min_uniform_uint32)

BINARY_OP_FUNC(__vec32_i64, __max_varying_int64, __max_uniform_int64)
BINARY_OP_FUNC(__vec32_i64, __min_varying_int64, __min_uniform_int64)
BINARY_OP_FUNC(__vec32_i64, __max_varying_uint64, __max_uniform_uint64)
BINARY_OP_FUNC(__vec32_i64, __min_varying_uint64, __min_uniform_uint64)

// sqrt/rsqrt/rcp

static FORCEINLINE float __rsqrt_uniform_float(float v) {
#ifdef ISPC_FAST_MATH
  return rsqrtf(v);
#else
  return 1.0f / sqrtf(v);
#endif
}

static FORCEINLINE float __rcp_uniform_float(float v) {
  return 1.0f/v;
}

static FORCEINLINE float __sqrt_uniform_float(float v) {
  return sqrtf(v);
}

static FORCEINLINE double __sqrt_uniform_double(double v) {
  return sqrt(v);
}

UNARY_OP(__vec32_f, __rcp_varying_float, __rcp_uniform_float)
UNARY_OP(__vec32_f, __rsqrt_varying_float, __rsqrt_uniform_float)
UNARY_OP(__vec32_f, __sqrt_varying_float, __sqrt_uniform_float)
UNARY_OP(__vec32_d, __sqrt_varying_double, __sqrt_uniform_double)

///////////////////////////////////////////////////////////////////////////
// bit ops

static FORCEINLINE int32_t __popcnt_int32(uint32_t v) {
  return __popc(v);
}
static FORCEINLINE int32_t __popcnt_int64(uint64_t v) {
  return __popcll(v);
}

static FORCEINLINE int32_t __count_trailing_zeros_i32(uint32_t v) {
  // ctz(x) =  popc((x&(-x))-1) from Wikipedia
  return __popc((v&(-v))-1);
}
static FORCEINLINE int64_t __count_trailing_zeros_i64(uint64_t v) {
  // ctz(x) =  popc((x&(-x))-1) from Wikipedia
  return __popcll((v&(-v))-1);
}

static FORCEINLINE int32_t __count_leading_zeros_i32(uint32_t v) {
  return __clz(v);
}
static FORCEINLINE int64_t __count_leading_zeros_i64(uint64_t v) {
  return __clzll(v);
}

///////////////////////////////////////////////////////////////////////////
// reductions

REDUCE_ADD(float, __vec32_f, __reduce_add_float)
REDUCE_MIN(float, __vec32_f, __reduce_min_float, <)
REDUCE_MAX(float, __vec32_f, __reduce_max_float, >)

REDUCE_ADD(double, __vec32_d, __reduce_add_double)
REDUCE_MIN(double, __vec32_d, __reduce_min_double, <)
REDUCE_MAX(double, __vec32_d, __reduce_max_double, >)

REDUCE_ADD(int16_t, __vec32_i8, __reduce_add_int8)
REDUCE_ADD(int32_t, __vec32_i16, __reduce_add_int16)

REDUCE_ADD(int64_t, __vec32_i32, __reduce_add_int32)
REDUCE_MIN(int32_t, __vec32_i32, __reduce_min_int32, <)
REDUCE_MAX(int32_t, __vec32_i32, __reduce_max_int32, >)

REDUCE_MIN(uint32_t, __vec32_i32, __reduce_min_uint32, <)
REDUCE_MAX(uint32_t, __vec32_i32, __reduce_max_uint32, >)

REDUCE_ADD(int64_t, __vec32_i64, __reduce_add_int64)
REDUCE_MIN(int64_t, __vec32_i64, __reduce_min_int64, <)
REDUCE_MAX(int64_t, __vec32_i64, __reduce_max_int64, >)

REDUCE_MIN(uint64_t, __vec32_i64, __reduce_min_uint64, <)
REDUCE_MAX(uint64_t, __vec32_i64, __reduce_max_uint64, >)

///////////////////////////////////////////////////////////////////////////
// masked load/store

static FORCEINLINE __vec32_i8 __masked_load_i8(void *p,
                                               __vec32_i1 mask) {
    __vec32_i8 ret;
    int8_t *ptr = (int8_t *)p;
    if ((mask.v) != 0)
            ret.v = ptr[programIndex()];
    return ret;
}

static FORCEINLINE __vec32_i16 __masked_load_i16(void *p,
                                                 __vec32_i1 mask) {
    __vec32_i16 ret;
    int16_t *ptr = (int16_t *)p;
    if ((mask.v ) != 0)
            ret.v = ptr[programIndex()];
    return ret;
}

static FORCEINLINE __vec32_i32 __masked_load_i32(void *p,
                                                 __vec32_i1 mask) {
    __vec32_i32 ret;
    int32_t *ptr = (int32_t *)p;
    if ((mask.v ) != 0)
            ret.v = ptr[programIndex()];
    return ret;
}

static FORCEINLINE __vec32_f __masked_load_float(void *p,
                                                 __vec32_i1 mask) {
    __vec32_f ret;
    float *ptr = (float *)p;
    if ((mask.v ) != 0)
            ret.v = ptr[programIndex()];
    return ret;
}

static FORCEINLINE __vec32_i64 __masked_load_i64(void *p,
                                                 __vec32_i1 mask) {
    __vec32_i64 ret;
    int64_t *ptr = (int64_t *)p;
    if ((mask.v) != 0)
            ret.v = ptr[programIndex()];
    return ret;
}

static FORCEINLINE __vec32_d __masked_load_double(void *p,
                                                  __vec32_i1 mask) {
    __vec32_d ret;
    double *ptr = (double *)p;
    if ((mask.v ) != 0)
      ret.v = ptr[programIndex()];
    return ret;
}

static FORCEINLINE void __masked_store_i8(void *p, __vec32_i8 val,
                                          __vec32_i1 mask) {
    int8_t *ptr = (int8_t *)p;
    if ((mask.v ) != 0)
      ptr[programIndex()] = val.v;
}

static FORCEINLINE void __masked_store_i16(void *p, __vec32_i16 val,
                                           __vec32_i1 mask) {
    int16_t *ptr = (int16_t *)p;
    if ((mask.v ) != 0)
      ptr[programIndex()] = val.v;
}

static FORCEINLINE void __masked_store_i32(void *p, __vec32_i32 val,
                                           __vec32_i1 mask) {
    int32_t *ptr = (int32_t *)p;
    if ((mask.v ) != 0)
      ptr[programIndex()] = val.v;
}

static FORCEINLINE void __masked_store_float(void *p, __vec32_f val,
                                             __vec32_i1 mask) {
    float *ptr = (float *)p;
    if ((mask.v ) != 0)
      ptr[programIndex()] = val.v;
}

static FORCEINLINE void __masked_store_i64(void *p, __vec32_i64 val,
                                          __vec32_i1 mask) {
    int64_t *ptr = (int64_t *)p;
    if ((mask.v ) != 0)
      ptr[programIndex()] = val.v;
}

static FORCEINLINE void __masked_store_double(void *p, __vec32_d val,
                                              __vec32_i1 mask) {
    double *ptr = (double *)p;
    if ((mask.v ) != 0)
      ptr[programIndex()] = val.v;
}

static FORCEINLINE void __masked_store_blend_i8(void *p, __vec32_i8 val,
                                                __vec32_i1 mask) {
    __masked_store_i8(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_i16(void *p, __vec32_i16 val,
                                                 __vec32_i1 mask) {
    __masked_store_i16(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_i32(void *p, __vec32_i32 val,
                                                 __vec32_i1 mask) {
    __masked_store_i32(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_float(void *p, __vec32_f val,
                                                   __vec32_i1 mask) {
    __masked_store_float(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_i64(void *p, __vec32_i64 val,
                                                 __vec32_i1 mask) {
    __masked_store_i64(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_double(void *p, __vec32_d val,
                                                    __vec32_i1 mask) {
    __masked_store_double(p, val, mask);
}

///////////////////////////////////////////////////////////////////////////
// gather/scatter

// offsets * offsetScale is in bytes (for all of these)

#define GATHER_BASE_OFFSETS(VTYPE, STYPE, OTYPE, FUNC)                  \
static FORCEINLINE VTYPE FUNC(unsigned char *b, uint32_t scale,         \
                              OTYPE offset, __vec32_i1 mask) {          \
    VTYPE ret;                                                          \
    int8_t *base = (int8_t *)b;                                         \
    if ((mask.v ) != 0) {                                               \
        STYPE *ptr = (STYPE *)(base + scale * offset.v);                \
        ret.v = *ptr;                                                   \
    }                                                                   \
    return ret;                                                         \
}
    

GATHER_BASE_OFFSETS(__vec32_i8,  int8_t,  __vec32_i32, __gather_base_offsets32_i8)
GATHER_BASE_OFFSETS(__vec32_i8,  int8_t,  __vec32_i64, __gather_base_offsets64_i8)
GATHER_BASE_OFFSETS(__vec32_i16, int16_t, __vec32_i32, __gather_base_offsets32_i16)
GATHER_BASE_OFFSETS(__vec32_i16, int16_t, __vec32_i64, __gather_base_offsets64_i16)
GATHER_BASE_OFFSETS(__vec32_i32, int32_t, __vec32_i32, __gather_base_offsets32_i32)
GATHER_BASE_OFFSETS(__vec32_i32, int32_t, __vec32_i64, __gather_base_offsets64_i32)
GATHER_BASE_OFFSETS(__vec32_f,   float,   __vec32_i32, __gather_base_offsets32_float)
GATHER_BASE_OFFSETS(__vec32_f,   float,   __vec32_i64, __gather_base_offsets64_float)
GATHER_BASE_OFFSETS(__vec32_i64, int64_t, __vec32_i32, __gather_base_offsets32_i64)
GATHER_BASE_OFFSETS(__vec32_i64, int64_t, __vec32_i64, __gather_base_offsets64_i64)
GATHER_BASE_OFFSETS(__vec32_d,   double,  __vec32_i32, __gather_base_offsets32_double)
GATHER_BASE_OFFSETS(__vec32_d,   double,  __vec32_i64, __gather_base_offsets64_double)

#define GATHER_GENERAL(VTYPE, STYPE, PTRTYPE, FUNC)         \
static FORCEINLINE VTYPE FUNC(PTRTYPE ptrs, __vec32_i1 mask) {   \
    VTYPE ret;                                              \
    if ((mask.v ) != 0) {                                   \
        STYPE *ptr = (STYPE *)ptrs.v;                       \
        ret.v = *ptr;                                       \
    }                                                       \
    return ret;                                             \
}

GATHER_GENERAL(__vec32_i8,  int8_t,  __vec32_i32, __gather32_i8)
GATHER_GENERAL(__vec32_i8,  int8_t,  __vec32_i64, __gather64_i8)
GATHER_GENERAL(__vec32_i16, int16_t, __vec32_i32, __gather32_i16)
GATHER_GENERAL(__vec32_i16, int16_t, __vec32_i64, __gather64_i16)
GATHER_GENERAL(__vec32_i32, int32_t, __vec32_i32, __gather32_i32)
GATHER_GENERAL(__vec32_i32, int32_t, __vec32_i64, __gather64_i32)
GATHER_GENERAL(__vec32_f,   float,   __vec32_i32, __gather32_float)
GATHER_GENERAL(__vec32_f,   float,   __vec32_i64, __gather64_float)
GATHER_GENERAL(__vec32_i64, int64_t, __vec32_i32, __gather32_i64)
GATHER_GENERAL(__vec32_i64, int64_t, __vec32_i64, __gather64_i64)
GATHER_GENERAL(__vec32_d,   double,  __vec32_i32, __gather32_double)
GATHER_GENERAL(__vec32_d,   double,  __vec32_i64, __gather64_double)

// scatter

#define SCATTER_BASE_OFFSETS(VTYPE, STYPE, OTYPE, FUNC)                 \
static FORCEINLINE void FUNC(unsigned char *b, uint32_t scale,          \
                             OTYPE offset, VTYPE val,                   \
                             __vec32_i1 mask) {                         \
    int8_t *base = (int8_t *)b;                                         \
    if ((mask.v ) != 0) {                                               \
        STYPE *ptr = (STYPE *)(base + scale * offset.v);                \
        *ptr = val.v;                                                   \
    }                                                                   \
}
    

SCATTER_BASE_OFFSETS(__vec32_i8,  int8_t,  __vec32_i32, __scatter_base_offsets32_i8)
SCATTER_BASE_OFFSETS(__vec32_i8,  int8_t,  __vec32_i64, __scatter_base_offsets64_i8)
SCATTER_BASE_OFFSETS(__vec32_i16, int16_t, __vec32_i32, __scatter_base_offsets32_i16)
SCATTER_BASE_OFFSETS(__vec32_i16, int16_t, __vec32_i64, __scatter_base_offsets64_i16)
SCATTER_BASE_OFFSETS(__vec32_i32, int32_t, __vec32_i32, __scatter_base_offsets32_i32)
SCATTER_BASE_OFFSETS(__vec32_i32, int32_t, __vec32_i64, __scatter_base_offsets64_i32)
SCATTER_BASE_OFFSETS(__vec32_f,   float,   __vec32_i32, __scatter_base_offsets32_float)
SCATTER_BASE_OFFSETS(__vec32_f,   float,   __vec32_i64, __scatter_base_offsets64_float)
SCATTER_BASE_OFFSETS(__vec32_i64, int64_t, __vec32_i32, __scatter_base_offsets32_i64)
SCATTER_BASE_OFFSETS(__vec32_i64, int64_t, __vec32_i64, __scatter_base_offsets64_i64)
SCATTER_BASE_OFFSETS(__vec32_d,   double,  __vec32_i32, __scatter_base_offsets32_double)
SCATTER_BASE_OFFSETS(__vec32_d,   double,  __vec32_i64, __scatter_base_offsets64_double)

#define SCATTER_GENERAL(VTYPE, STYPE, PTRTYPE, FUNC)                 \
static FORCEINLINE void FUNC(PTRTYPE ptrs, VTYPE val, __vec32_i1 mask) {  \
    VTYPE ret;                                                       \
    if ((mask.v ) != 0) {                                            \
        STYPE *ptr = (STYPE *)ptrs.v;                                \
        *ptr = val.v;                                                \
    }                                                                \
}

SCATTER_GENERAL(__vec32_i8,  int8_t,  __vec32_i32, __scatter32_i8)
SCATTER_GENERAL(__vec32_i8,  int8_t,  __vec32_i64, __scatter64_i8)
SCATTER_GENERAL(__vec32_i16, int16_t, __vec32_i32, __scatter32_i16)
SCATTER_GENERAL(__vec32_i16, int16_t, __vec32_i64, __scatter64_i16)
SCATTER_GENERAL(__vec32_i32, int32_t, __vec32_i32, __scatter32_i32)
SCATTER_GENERAL(__vec32_i32, int32_t, __vec32_i64, __scatter64_i32)
SCATTER_GENERAL(__vec32_f,   float,   __vec32_i32, __scatter32_float)
SCATTER_GENERAL(__vec32_f,   float,   __vec32_i64, __scatter64_float)
SCATTER_GENERAL(__vec32_i64, int64_t, __vec32_i32, __scatter32_i64)
SCATTER_GENERAL(__vec32_i64, int64_t, __vec32_i64, __scatter64_i64)
SCATTER_GENERAL(__vec32_d,   double,  __vec32_i32, __scatter32_double)
SCATTER_GENERAL(__vec32_d,   double,  __vec32_i64, __scatter64_double)

///////////////////////////////////////////////////////////////////////////
// packed load/store

static FORCEINLINE int32_t __packed_load_active(int32_t *ptr, __vec32_i32 *val, __vec32_i1 mask) 
{
  int count = 0; 
  for (int i = 0; i < 32; i++)
  {
    const bool cond = mask.v && programIndex() == i;
    if (cond)
      val->v = *(ptr+count);
    count += __ballot(cond) != 0;
  }
  return count;
}


static FORCEINLINE int32_t __packed_store_active(int32_t *ptr, __vec32_i32 val,  __vec32_i1 mask) 
{
  int count = 0; 
  for (int i = 0; i < 32; i++) 
  {
    const bool cond = mask.v && programIndex() == i;
    if (cond)
      *(ptr+count) = val.v;
    count += __ballot(cond) != 0;
  }
  return count;
}


static FORCEINLINE int32_t __packed_store_active2(int32_t *ptr, __vec32_i32 val, __vec32_i1 mask) 
{
#if 0
  int count = 0;
  int32_t *ptr_ = ptr;
  for (int i = 0; i < 16; ++i) {
    *ptr = val.v[i];
    ptr += mask.v & 1;
    mask.v = mask.v >> 1;
  }
  return ptr - ptr_;
#else
  return __packed_store_active(ptr, val, mask);
#endif
}


static FORCEINLINE int32_t __packed_load_active(uint32_t *ptr,
                                                __vec32_i32 *val,
                                                __vec32_i1 mask) {
    return __packed_load_active((int32_t *)ptr, val, mask);
}


static FORCEINLINE int32_t __packed_store_active(uint32_t *ptr, 
                                                 __vec32_i32 val,
                                                 __vec32_i1 mask) {
    return __packed_store_active((int32_t *)ptr, val, mask);
}


static FORCEINLINE int32_t __packed_store_active2(uint32_t *ptr,
                                                 __vec32_i32 val,
                                                 __vec32_i1 mask) {
    return __packed_store_active2((int32_t *)ptr, val, mask);
}


///////////////////////////////////////////////////////////////////////////
// aos/soa

static FORCEINLINE void __soa_to_aos3_float(__vec32_f v0, __vec32_f v1, __vec32_f v2, float *ptr) {
  int count = 0;
  for (int i = 0; i < 32; ++i) 
  {
    const bool cond = programIndex() == i;
    if (cond)
    {
      *(ptr+3*count+0) = v0.v;
      *(ptr+3*count+1) = v1.v;
      *(ptr+3*count+2) = v2.v; 
    }
    count += __ballot(cond) != 0;
  }
}

static FORCEINLINE void __aos_to_soa3_float(float *ptr, __vec32_f *out0, __vec32_f *out1, __vec32_f *out2) {
  int count = 0;
  for (int i = 0; i < 32; ++i)
  {
    const bool cond = programIndex() == i;
    if (cond)
    {
      out0->v = *(ptr + 3*count + 0);
      out1->v = *(ptr + 3*count + 1);
      out2->v = *(ptr + 3*count + 2);
    }
    count += __ballot(cond) != 0;
  }
}

static FORCEINLINE void __soa_to_aos4_float(__vec32_f v0, __vec32_f v1, __vec32_f v2,
                                            __vec32_f v3, float *ptr) {
  int count = 0;
  for (int i = 0; i < 32; ++i) 
  {
    const bool cond = programIndex() == i;
    if (cond)
    {
      *(ptr+4*count+0) = v0.v;
      *(ptr+4*count+1) = v1.v;
      *(ptr+4*count+2) = v2.v; 
      *(ptr+4*count+3) = v3.v; 
    }
    count += __ballot(cond) != 0;
  }
}

static FORCEINLINE void __aos_to_soa4_float(float *ptr, __vec32_f *out0, __vec32_f *out1,
                                            __vec32_f *out2, __vec32_f *out3) {
  int count = 0;
  for (int i = 0; i < 32; ++i)
  {
    const bool cond = programIndex() == i;
    if (cond)
    {
      out0->v = *(ptr + 4*count + 0);
      out1->v = *(ptr + 4*count + 1);
      out2->v = *(ptr + 4*count + 2);
      out3->v = *(ptr + 4*count + 3);
    }
    count += __ballot(cond) != 0;
  }
}

///////////////////////////////////////////////////////////////////////////
// prefetch

static FORCEINLINE void __prefetch_read_uniform_1(unsigned char *) {
}

static FORCEINLINE void __prefetch_read_uniform_2(unsigned char *) {
}

static FORCEINLINE void __prefetch_read_uniform_3(unsigned char *) {
}

static FORCEINLINE void __prefetch_read_uniform_nt(unsigned char *) {
}

///////////////////////////////////////////////////////////////////////////
// atomics

static FORCEINLINE uint32_t __atomic_add(uint32_t *p, uint32_t v) {
  return atomicAdd(p, v);
}

static FORCEINLINE uint32_t __atomic_sub(uint32_t *p, uint32_t v) {
  return atomicSub(p, v);
}

static FORCEINLINE uint32_t __atomic_and(uint32_t *p, uint32_t v) {
  return atomicAnd(p, v);
}

static FORCEINLINE uint32_t __atomic_or(uint32_t *p, uint32_t v) {
  return atomicOr(p, v);
}

static FORCEINLINE uint32_t __atomic_xor(uint32_t *p, uint32_t v) {
  return atomicXor(p, v);
}

static FORCEINLINE int32_t __atomic_min(uint32_t *p, uint32_t v) {
  return atomicMin((int32_t*)p, (int32_t)v);
}

static FORCEINLINE int32_t __atomic_max(uint32_t *p, uint32_t v) {
  return atomicMax((int32_t*)p, (int32_t)v);
}

static FORCEINLINE uint32_t __atomic_umin(uint32_t *p, uint32_t v) {
  return atomicMin(p, v);
}

static FORCEINLINE uint32_t __atomic_umax(uint32_t *p, uint32_t v) {
  return atomicMax(p, v);
}

static FORCEINLINE uint32_t __atomic_xchg(uint32_t *p, uint32_t v) {
  return atomicExch(p, v);
}

static FORCEINLINE uint32_t __atomic_cmpxchg(uint32_t *p, uint32_t cmpval,
                                             uint32_t newval) {
  return atomicCAS(p, cmpval, newval);
}

static FORCEINLINE uint64_t __atomic_add(uint64_t *p, uint64_t v) {
  return atomicAdd(p, v);
}

static FORCEINLINE uint64_t __atomic_sub(uint64_t *p, uint64_t v) {
  return atomicAdd(p, -v);
}

static FORCEINLINE uint64_t __atomic_and(uint64_t *p, uint64_t v) {
  return atomicAnd(p, v);
}

static FORCEINLINE uint64_t __atomic_or(uint64_t *p, uint64_t v) {
  return atomicOr(p, v);
}

static FORCEINLINE uint64_t __atomic_xor(uint64_t *p, uint64_t v) {
  return atomicXor(p, v);
}

static FORCEINLINE uint64_t __atomic_min(uint64_t *p, uint64_t v) {
  int64_t old, min;
  do {
    old = *((volatile int64_t *)p);
    min = (old < (int64_t)v) ? old : (int64_t)v;
  } while (atomicCAS(p, old, min) != old);
  return old;
}

static FORCEINLINE uint64_t __atomic_max(uint64_t *p, uint64_t v) {
  int64_t old, max;
  do {
    old = *((volatile int64_t *)p);
    max = (old > (int64_t)v) ? old : (int64_t)v;
  } while (atomicCAS(p, old, max) != old);
  return old;
}

static FORCEINLINE uint64_t __atomic_umin(uint64_t *p, uint64_t v) {
  return atomicMin(p, v);
}

static FORCEINLINE uint64_t __atomic_umax(uint64_t *p, uint64_t v) {
  return atomicMax(p, v);
}

static FORCEINLINE uint64_t __atomic_xchg(uint64_t *p, uint64_t v) {
  return atomicExch(p, v);
}

static FORCEINLINE uint64_t __atomic_cmpxchg(uint64_t *p, uint64_t cmpval,
                                             uint64_t newval) {
  return atomicCAS(p, cmpval, newval);
}


#if 0
static FORCEINLINE uint64_t __clock() {
  uint32_t low, high;
#ifdef __x86_64
  __asm__ __volatile__ ("xorl %%eax,%%eax \n    cpuid"
                        ::: "%rax", "%rbx", "%rcx", "%rdx" );
#else
  __asm__ __volatile__ ("xorl %%eax,%%eax \n    cpuid"
                        ::: "%eax", "%ebx", "%ecx", "%edx" );
#endif
  __asm__ __volatile__ ("rdtsc" : "=a" (low), "=d" (high));
  return (uint64_t)high << 32 | low;
}
#endif

/*** new/delete ****/

static FORCEINLINE uint8_t* __new_uniform_64rt(uint64_t size)
{
  uint8_t* ptr;
  if (programIndex() == 0)
    ptr = static_cast<uint8_t*>(malloc(size));
  uint64_t ptr_i64 = (uint64_t)ptr;
  ptr_i64 = __shuffle<uint64_t>(ptr_i64,0);
  ptr = (uint8_t*)ptr_i64;
  return ptr;
}
static FORCEINLINE void __delete_uniform_64rt(uint8_t *ptr)
{
  if (programIndex() == 0)
    delete ptr;
}
static FORCEINLINE __vec32_i64 __new_varying32_64rt(__vec32_i32 size,  __vec32_i1 mask)
{
  __vec32_i64 ptr_i64(0);
  if (mask.v)
  {
    uint8_t* ptr = (uint8_t*)malloc(size.v);
    ptr_i64.v = (uint64_t)ptr;
  }
  return ptr_i64;
}
static FORCEINLINE __vec32_i64 __new_varying64_64rt(__vec32_i64 size,  __vec32_i1 mask)
{
  __vec32_i64 ptr_i64(0);
  if (mask.v)
  {
    uint8_t* ptr = (uint8_t*)malloc(size.v);
    ptr_i64.v = (uint64_t)ptr;
  }
  return ptr_i64;
}
static FORCEINLINE void __delete_varying_64rt(__vec32_i64 ptr_i64, __vec32_i1 mask)
{
  if (mask.v)
  {
    uint8_t* ptr = (uint8_t*)ptr_i64.v;
    delete(ptr);
  }
}


