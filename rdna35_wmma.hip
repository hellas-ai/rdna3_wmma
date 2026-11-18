// Standalone RDNA3.5 WMMA arithmetic model and differential validator.
// No Python, PyTorch, rocBLAS, or repository libraries are used.
#define WMMA_CUDA 0
#ifndef WMMA_CPU_ONLY
#if defined(__CUDACC__) && !defined(__HIPCC__)
#undef WMMA_CUDA
#define WMMA_CUDA 1
#include <cuda_runtime.h>
// Keep the existing HIP call sites; direct nvcc builds need no HIP installation.
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipGetDeviceCount cudaGetDeviceCount
#define hipDeviceProp_t cudaDeviceProp
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipSetDevice cudaSetDevice
#define hipStream_t cudaStream_t
#define hipStreamCreateWithFlags cudaStreamCreateWithFlags
#define hipStreamNonBlocking cudaStreamNonBlocking
#define hipStreamDestroy cudaStreamDestroy
#define hipStreamSynchronize cudaStreamSynchronize
#define hipMemcpyAsync cudaMemcpyAsync
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipGetLastError cudaGetLastError
#define hipEvent_t cudaEvent_t
#define hipEventCreate cudaEventCreate
#define hipEventDestroy cudaEventDestroy
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEventElapsedTime cudaEventElapsedTime
#else
#include <hip/hip_runtime.h>
#endif
#define HD __host__ __device__
#else
#define HD
#endif
#if !defined(WMMA_CPU_ONLY) && !WMMA_CUDA
#define WMMA_AMD_GPU 1
#else
#define WMMA_AMD_GPU 0
#endif
#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifndef WMMA_SOURCE_SHA256
#define WMMA_SOURCE_SHA256 "not recorded (build with the supplied Makefile)"
#endif

namespace rdna35 {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "FP32 host reference required");
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::digits == 53, "FP64 host reference required");
enum class Format { BF16, FP16 };
constexpr uint32_t QNAN = 0x7fc00000u;
constexpr uint32_t OPERAND_NAN = 0xffc00a3du;
constexpr uint32_t INVALID_NAN = 0xffc00000u;
constexpr uint32_t INF = 0x7f800000u;
struct Decoded {
    uint32_t significand;
    int exponent; // value = significand * 2**(exponent - fraction_bits)
    bool sign, infinity, nan;
};
template<Format F> HD Decoded decode16(uint16_t bits) {
    constexpr int fraction = F == Format::BF16 ? 7 : 10;
    constexpr int bias = F == Format::BF16 ? 127 : 15;
    constexpr int exp_mask = F == Format::BF16 ? 255 : 31;
    const int e = (bits >> fraction) & exp_mask;
    const uint32_t m = bits & ((1u << fraction) - 1);
    // BF16 subnormals flush to signed zero; FP16 subnormals are retained.
    const uint32_t sig = e ? ((1u << fraction) | m) : (F == Format::BF16 ? 0u : m);
    return {sig, e ? e - bias : 1 - bias, bool(bits & 0x8000),
            e == exp_mask && m == 0, e == exp_mask && m != 0};
}
HD Decoded decode32(uint32_t bits) {
    const int e = (bits >> 23) & 255;
    const uint32_t m = bits & 0x7fffff;
    return {e ? (0x800000u | m) : 0u, e ? e - 127 : -126,
            bool(bits >> 31), e == 255 && m == 0, e == 255 && m != 0};
}
HD uint64_t align_unsigned(uint64_t significand, int shift) {
    if (!significand) return 0;
    if (shift >= 0) return significand << shift; // bounded by 27 bits here
    return shift <= -64 ? 0 : significand >> -shift;
}
HD int64_t align_signed(uint64_t significand, int shift, bool sign) {
    const int64_t aligned = static_cast<int64_t>(align_unsigned(significand, shift));
    return sign ? -aligned : aligned; // truncation toward zero
}
HD int msb(uint64_t x) { // x != 0
#ifdef __CUDA_ARCH__
    return 63 - __clzll(static_cast<unsigned long long>(x));
#else
    return 63 - __builtin_clzll(x);
#endif
}

// Convert S * 2**power to raw FP32, with explicit RNE and measured underflow.
HD uint32_t round_result(int64_t s, int power) {
    if (!s) return 0; // arithmetic zeros always +0
    const bool negative = s < 0;
    const uint64_t magnitude = negative ? uint64_t(-s) : uint64_t(s);
    const uint32_t sign = negative ? 0x80000000u : 0;
    const int high = msb(magnitude);
    int exponent = high + power;
    if (exponent < -126) {
        if (exponent < -127) return 0;
        // Compare exactly with (2**25 - 1) * 2**-151.
        // E >= -128, hence power+151 >= -1. At exponent -127
        // both sides of this comparison fit within 27 bits.
        const int shift = power + 151;
        const uint64_t threshold = (uint64_t(1) << 25) - 1;
        const bool below = shift >= 0 ? (magnitude << shift) < threshold
                                      : magnitude < (threshold << -shift);
        return below ? 0 : sign | 0x00800000u;
    }
    const int shift = high - 23;
    uint64_t mantissa;
    if (shift > 0) {
        mantissa = magnitude >> shift;
        const uint64_t remainder = magnitude & ((uint64_t(1) << shift) - 1);
        const uint64_t half = uint64_t(1) << (shift - 1);
        if (remainder > half || (remainder == half && (mantissa & 1))) ++mantissa;
    } else {
        mantissa = magnitude << -shift;
    }
    if (mantissa == (uint64_t(1) << 24)) { mantissa >>= 1; ++exponent; }
    if (exponent > 127) return sign | INF;
    return sign | (uint32_t(exponent + 127) << 23) | (uint32_t(mantissa) & 0x7fffff);
}

template<Format F>
HD uint32_t simulate_dot2(uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1, uint32_t cbits) {
    constexpr int fraction = F == Format::BF16 ? 7 : 10;
    const Decoded a[2] = {decode16<F>(a0), decode16<F>(a1)};
    const Decoded b[2] = {decode16<F>(b0), decode16<F>(b1)};
    const Decoded c = decode32(cbits);
    bool positive_inf = c.infinity && !c.sign;
    bool negative_inf = c.infinity && c.sign;
    if (a[0].nan || a[1].nan || b[0].nan || b[1].nan) return OPERAND_NAN;
    uint64_t product[2];
    bool sign[2];
    int ep[2];
    int e = c.exponent - 2;
    for (int i = 0; i < 2; ++i) {
        sign[i] = a[i].sign != b[i].sign;
        const bool invalid = (a[i].infinity && !b[i].significand && !b[i].infinity) ||
                             (b[i].infinity && !a[i].significand && !a[i].infinity);
        if (invalid) return INVALID_NAN;
        if (a[i].infinity || b[i].infinity) {
            positive_inf |= !sign[i]; negative_inf |= sign[i];
        }
        product[i] = uint64_t(a[i].significand) * b[i].significand;
        ep[i] = a[i].exponent + b[i].exponent; // deliberately NOT normalized
        if (product[i] && ep[i] > e) e = ep[i];
    }
    if (positive_inf && negative_inf) return INVALID_NAN;
    if (c.nan) return cbits | 0x00400000u;
    if (positive_inf || negative_inf) return (negative_inf ? 0x80000000u : 0) | INF;
    int64_t sum = align_signed(c.significand, c.exponent + 1 - e, c.sign);
    for (int i = 0; i < 2; ++i)
        sum += align_signed(product[i], ep[i] - 2 * fraction + 24 - e, sign[i]);
    sum -= int(sign[0]) + int(sign[1]); // includes signed zero products
    sum += 2 * int(c.sign);           // includes a signed zero accumulator
    return round_result(sum, e - 24);
}

template<Format F>
HD uint32_t simulate_output(const uint16_t* a, const uint16_t* bt, uint32_t c) {
    for (int k = 0; k < 16; k += 2)
        c = simulate_dot2<F>(a[k], bt[k], a[k + 1], bt[k + 1], c);
    return c;
}

// Matrices: A row-major, B column-major (bt[n*16+k] == B[k,n]),
// C and D row-major. FP32 C/D are passed as raw uint32_t encodings.
void simulate_wmma_bf16_cpu(const uint16_t* a, const uint16_t* bt,
                            const uint32_t* c, uint32_t* d) {
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n)
            d[m * 16 + n] = simulate_output<Format::BF16>(a + m * 16, bt + n * 16, c[m * 16 + n]);
}
void simulate_wmma_f16_cpu(const uint16_t* a, const uint16_t* bt,
                           const uint32_t* c, uint32_t* d) {
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n)
            d[m * 16 + n] = simulate_output<Format::FP16>(a + m * 16, bt + n * 16, c[m * 16 + n]);
}

#ifndef WMMA_CPU_ONLY
__global__ void simulate_wmma_bf16_gpu(const uint16_t* a, const uint16_t* bt,
                                      const uint32_t* c, uint32_t* d, int tiles) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tiles * 256) return;
    const int base = (i / 256) * 256, cell = i % 256;
    d[i] = simulate_output<Format::BF16>(a + base + (cell / 16) * 16,
                                        bt + base + (cell % 16) * 16, c[i]);
}
__global__ void simulate_wmma_f16_gpu(const uint16_t* a, const uint16_t* bt,
                                     const uint32_t* c, uint32_t* d, int tiles) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tiles * 256) return;
    const int base = (i / 256) * 256, cell = i % 256;
    d[i] = simulate_output<Format::FP16>(a + base + (cell / 16) * 16,
                                       bt + base + (cell % 16) * 16, c[i]);
}
#if WMMA_AMD_GPU
using Float8 = float __attribute__((ext_vector_type(8)));
using Short16 = short __attribute__((ext_vector_type(16)));
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
__device__ float float_from_bits(uint32_t u) { union { uint32_t u; float f; } x = {u}; return x.f; }
__device__ uint32_t float_to_bits(float f) { union { float f; uint32_t u; } x = {f}; return x.u; }
__device__ _Float16 half_from_bits(uint16_t u) { union { uint16_t u; _Float16 f; } x = {u}; return x.f; }

// Independent hardware oracle: one real wave32 WMMA instruction per tile.
template<Format F>
__global__ void hardware_wmma(const uint16_t* a, const uint16_t* bt,
                              const uint32_t* c, uint32_t* d) {
    const int base = blockIdx.x * 256, lane = threadIdx.x;
    const int column = lane % 16, parity = lane / 16;
    Float8 cf, df;
    for (int i = 0; i < 8; ++i) cf[i] = float_from_bits(c[base + (parity + 2 * i) * 16 + column]);
    if constexpr (F == Format::BF16) {
        Short16 af, bf;
        for (int k = 0; k < 16; ++k) {
            af[k] = static_cast<short>(a[base + column * 16 + k]);
            bf[k] = static_cast<short>(bt[base + column * 16 + k]);
        }
        df = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf, cf);
    } else {
        Half16 af, bf;
        for (int k = 0; k < 16; ++k) {
            af[k] = half_from_bits(a[base + column * 16 + k]);
            bf[k] = half_from_bits(bt[base + column * 16 + k]);
        }
        df = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, cf);
    }
    for (int i = 0; i < 8; ++i) d[base + (parity + 2 * i) * 16 + column] = float_to_bits(df[i]);
}
// Native dot2 arithmetic is deliberately separate from the integer emulator.
template<Format F> __device__ float native_dot2(uint32_t a, uint32_t b, float c) {
    float d;
    if constexpr (F == Format::BF16)
        asm volatile("v_dot2_f32_bf16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    else
        asm volatile("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
}
__device__ uint32_t pack_pair(const uint16_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 16); }

template<Format F>
__global__ void native_dot2_outputs(const uint16_t* a, const uint16_t* bt, const uint32_t* c,
                                   uint32_t* d, int tiles, int iterations) {
    // Eight waves per tile; each lane computes one complete output.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tiles * 256) return;
    const int base = (i / 256) * 256, cell = i % 256;
    uint32_t ap[8], bp[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        ap[k] = pack_pair(a + base + (cell / 16) * 16 + 2*k);
        bp[k] = pack_pair(bt + base + (cell % 16) * 16 + 2*k);
    }
    float value = float_from_bits(c[i]);
    for (int repeat = 0; repeat < iterations; ++repeat) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) value = native_dot2<F>(ap[k], bp[k], value);
    }
    d[i] = float_to_bits(value);
}

template<Format F>
__global__ void native_dot2_wave(const uint16_t* a, const uint16_t* bt, const uint32_t* c,
                                uint32_t* d, int iterations) {
    // One wave per tile, like WMMA. Gather each lane's eight A rows once,
    // outside the timed inner recurrence; this costs extra registers/shuffles.
    const int base = blockIdx.x * 256, lane = threadIdx.x;
    const int column = lane % 16, parity = lane / 16;
    uint32_t own_a[8], bp[8], gathered_a[8][8];
    float value[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        own_a[k] = pack_pair(a + base + column*16 + 2*k);
        bp[k] = pack_pair(bt + base + column*16 + 2*k);
    }
    #pragma unroll
    for (int row = 0; row < 8; ++row) {
        value[row] = float_from_bits(c[base + (parity + 2*row)*16 + column]);
        #pragma unroll
        for (int k = 0; k < 8; ++k) gathered_a[row][k] = __shfl(own_a[k], parity + 2*row, 32);
    }
    for (int repeat = 0; repeat < iterations; ++repeat) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            #pragma unroll
            for (int row = 0; row < 8; ++row) value[row] = native_dot2<F>(gathered_a[row][k], bp[k], value[row]);
        }
    }
    #pragma unroll
    for (int row = 0; row < 8; ++row) d[base + (parity + 2*row)*16 + column] = float_to_bits(value[row]);
}

template<Format F>
__global__ void benchmark_wmma(const uint16_t* a, const uint16_t* bt, const uint32_t* c,
                               uint32_t* d, int iterations) {
    const int base = blockIdx.x * 256, lane = threadIdx.x;
    const int column = lane % 16, parity = lane / 16;
    Float8 value;
    for (int row = 0; row < 8; ++row) value[row] = float_from_bits(c[base + (parity+2*row)*16+column]);
    if constexpr (F == Format::BF16) {
        Short16 af, bf;
        for (int k = 0; k < 16; ++k) { af[k] = short(a[base+column*16+k]); bf[k] = short(bt[base+column*16+k]); }
        for (int repeat = 0; repeat < iterations; ++repeat)
            value = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af,bf,value);
    } else {
        Half16 af, bf;
        for (int k = 0; k < 16; ++k) { af[k] = half_from_bits(a[base+column*16+k]); bf[k] = half_from_bits(bt[base+column*16+k]); }
        for (int repeat = 0; repeat < iterations; ++repeat)
            value = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af,bf,value);
    }
    for (int row = 0; row < 8; ++row) d[base+(parity+2*row)*16+column] = float_to_bits(value[row]);
}

#endif // WMMA_AMD_GPU

template<Format F>
__global__ void benchmark_integer(const uint16_t* a, const uint16_t* bt, const uint32_t* c,
                                  uint32_t* d, int tiles, int iterations) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tiles * 256) return;
    const int base = (i / 256) * 256, cell = i % 256;
    uint16_t av[16], bv[16];
    for (int k = 0; k < 16; ++k) { av[k] = a[base+(cell/16)*16+k]; bv[k] = bt[base+(cell%16)*16+k]; }
    uint32_t value = c[i];
    for (int repeat = 0; repeat < iterations; ++repeat) value = simulate_output<F>(av,bv,value);
    d[i] = value;
}
#endif

// Separate floating-point implementation, used only as a CPU cross-check.
// It shares the mathematical model, not the integer alignment/rounding code.
float as_float(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
uint32_t as_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
template<Format F> double as_double(uint16_t bits) {
    const auto x = decode16<F>(bits);
    if (x.nan) return std::numeric_limits<double>::quiet_NaN();
    if (x.infinity) return x.sign ? -INFINITY : INFINITY;
    constexpr int fraction = F == Format::BF16 ? 7 : 10;
    return std::copysign(std::ldexp(double(x.significand), x.exponent - fraction), x.sign ? -1. : 1.);
}
template<Format F>
uint32_t reference_output(const uint16_t* a, const uint16_t* b, uint32_t cbits) {
    for (int k = 0; k < 16; k += 2) {
        const double av[2] = {as_double<F>(a[k]), as_double<F>(a[k + 1])};
        const double bv[2] = {as_double<F>(b[k]), as_double<F>(b[k + 1])};
        const double p[2] = {av[0] * bv[0], av[1] * bv[1]};
        const Decoded cd = decode32(cbits);
        const double c = cd.nan ? NAN : cd.infinity ? (cd.sign ? -INFINITY : INFINITY) :
            std::copysign(std::ldexp(double(cd.significand), cd.exponent - 23), cd.sign ? -1. : 1.);
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(c)) {
            const bool input_nan = decode16<F>(a[k]).nan || decode16<F>(a[k+1]).nan ||
                                   decode16<F>(b[k]).nan || decode16<F>(b[k+1]).nan;
            if (input_nan) cbits = OPERAND_NAN;
            else if (std::isnan(p[0]) || std::isnan(p[1]) || std::isnan(p[0]+p[1])) cbits = INVALID_NAN;
            else if (cd.nan) cbits |= 0x00400000u;
            else if (std::isnan((p[0]+p[1])+c)) cbits = INVALID_NAN;
            else cbits = as_bits(float((p[0]+p[1])+c));
            continue;
        }
        int e = cd.exponent - 2;
        for (int j = 0; j < 2; ++j) {
            if (p[j] != 0) {
                const auto da = decode16<F>(a[k + j]), db = decode16<F>(b[k + j]);
                e = std::max(e, da.exponent + db.exponent);
            }
        }
        const double scale = std::ldexp(1., 24 - e);
        double s = std::trunc(p[0] * scale) + std::trunc(p[1] * scale) + std::trunc(c * scale);
        s -= int(std::signbit(p[0])) + int(std::signbit(p[1]));
        s += 2 * int(std::signbit(c));
        double result = s / scale;
        if (std::abs(result) < std::ldexp(1., -126) - std::ldexp(1., -151)) result = 0.;
        cbits = as_bits(float(result));
    }
    return cbits;
}

bool is_nan(uint32_t x) { return (x & 0x7fffffffu) > INF; }
bool equal(uint32_t a, uint32_t b, bool strict_nan) {
    return a == b || (!strict_nan && is_nan(a) && is_nan(b));
}
const char* name(Format f) { return f == Format::BF16 ? "bf16" : "f16"; }
struct Tile {
    std::array<uint16_t, 256> a{}, bt{};
    std::array<uint32_t, 256> c{};
};
struct RNG {
    uint64_t state;
    uint64_t next() { // specified SplitMix64: reproducible raw bits across platforms
        uint64_t z = (state += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    int range(int low, int high) { return low + int(next() % uint64_t(high - low + 1)); }
};
uint16_t encode16(Format f, bool sign, int exponent, uint32_t mantissa = 0) {
    const int fraction = f == Format::BF16 ? 7 : 10, bias = f == Format::BF16 ? 127 : 15;
    const int max_exp = f == Format::BF16 ? 127 : 15;
    exponent = std::max(1 - bias, std::min(max_exp, exponent));
    return uint16_t((sign ? 0x8000 : 0) | ((exponent + bias) << fraction) |
                    (mantissa & ((1 << fraction) - 1)));
}
uint32_t encode32(bool sign, int exponent, uint32_t mantissa = 0) {
    exponent = std::max(-126, std::min(127, exponent));
    return (sign ? 0x80000000u : 0) | (uint32_t(exponent + 127) << 23) | (mantissa & 0x7fffff);
}
Tile random_tile(Format f, const std::string& suite, RNG& rng) {
    Tile tile;
    const int targets[] = {-150, -127, -126, -125, -1, 0, 1, 126, 127, 128, 129};
    const int offsets[] = {-25, -24, -23, -1, 0, 1, 23};
    const int low = f == Format::BF16 ? -126 : -14, high = f == Format::BF16 ? 127 : 15;
    const int target = targets[rng.range(0, 10)], ea = rng.range(low, high);
    const int eb = std::max(low, std::min(high, target - ea));
    const int ec = target + offsets[rng.range(0, 6)];
    for (int i = 0; i < 256; ++i) {
        if (suite == "bits") {
            tile.a[i] = uint16_t(rng.next()); tile.bt[i] = uint16_t(rng.next()); tile.c[i] = uint32_t(rng.next());
        } else {
            const bool extreme = suite == "exponents";
            // Sequence RNG draws explicitly: C++ argument evaluation order
            // otherwise makes GCC and Clang generate different input corpora.
            const bool as = rng.next() & 1;
            const int ae = extreme ? ea : rng.range(-4, 3);
            const uint32_t am = uint32_t(rng.next());
            const bool bs = rng.next() & 1;
            const int be = extreme ? eb : rng.range(-4, 3);
            const uint32_t bm = uint32_t(rng.next());
            const bool cs = rng.next() & 1;
            const int ce = extreme ? ec : rng.range(-4, 3);
            const uint32_t cm = uint32_t(rng.next());
            tile.a[i] = encode16(f, as, ae, am);
            tile.bt[i] = encode16(f, bs, be, bm);
            tile.c[i] = encode32(cs, ce, cm);
        }
    }
    if (suite == "cancellation") {
        tile.c.fill(0);
        for (int i = 0; i < 256; i += 2) {
            tile.a[i + 1] = tile.a[i] ^ 0x8000;
            tile.bt[i + 1] = tile.bt[i];
        }
    }
    return tile;
}

struct Options {
    uint64_t trials = 1000, seed = 20261028;
    int batch = 128, threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    bool cpu_only = false, reference = true, strict_nan = true;
    bool benchmark = false, benchmark_only = false;
    int bench_tiles = 2048, bench_inner = 32, bench_repeats = 5, bench_samples = 5;
    std::string dtype = "both", report, failure = "rdna35_wmma_failure.txt", replay;
};
uint64_t number(const std::string& s) {
    if (s.empty() || s.front() == '-') throw std::runtime_error("expected a nonnegative integer: " + s);
    size_t n = 0; const auto v = std::stoull(s, &n, 0);
    if (n != s.size()) throw std::runtime_error("invalid integer: " + s);
    return v;
}
Options options(int argc, char** argv) {
    Options o;
#ifdef WMMA_CPU_ONLY
    o.cpu_only = true;
#endif
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() { if (++i >= argc) throw std::runtime_error("missing value for " + arg); return std::string(argv[i]); };
        if (arg == "--trials") o.trials = number(value());
        else if (arg == "--seed") o.seed = number(value());
        else if (arg == "--batch") { const auto n = number(value()); if (n < 1 || n > 4096) throw std::runtime_error("batch must be 1..4096"); o.batch = int(n); }
        else if (arg == "--threads") { const auto n = number(value()); if (n < 1 || n > 256) throw std::runtime_error("threads must be 1..256"); o.threads = int(n); }
        else if (arg == "--dtype") o.dtype = value();
        else if (arg == "--report") o.report = value();
        else if (arg == "--failure") o.failure = value();
        else if (arg == "--replay") o.replay = value();
        else if (arg == "--cpu-only") o.cpu_only = true;
        else if (arg == "--no-reference") o.reference = false;
        else if (arg == "--strict-nan") o.strict_nan = true;
        else if (arg == "--nan-classification") o.strict_nan = false;
        else if (arg == "--benchmark") o.benchmark = true;
        else if (arg == "--benchmark-only") { o.benchmark = true; o.benchmark_only = true; }
        else if (arg == "--bench-tiles" || arg == "--bench-inner" || arg == "--bench-repeats" || arg == "--bench-samples") {
            const auto n = number(value());
            const auto limit = arg == "--bench-tiles" ? 65536u : arg == "--bench-inner" ? 4096u : 1000u;
            if (n < 1 || n > limit) throw std::runtime_error("benchmark argument out of range: " + arg);
            if (arg == "--bench-tiles") o.bench_tiles = int(n);
            else if (arg == "--bench-inner") o.bench_inner = int(n);
            else if (arg == "--bench-repeats") o.bench_repeats = int(n);
            else o.bench_samples = int(n);
        }
        else if (arg == "--help") {
            std::cout << "RDNA3.5 WMMA standalone validator\n"
                      << "  --trials N       random tiles per family and dtype (default 1000)\n"
                      << "  --seed N         SplitMix64 seed (default 20261028)\n"
                      << "  --batch N        tiles per GPU batch, 1..4096 (default 128)\n"
                      << "  --threads N      CPU worker threads (default up to 8)\n"
                      << "  --dtype TYPE     bf16, f16, or both (default both)\n"
                      << "  --cpu-only       run CPU integer versus FP64 reference; no GPU access\n"
                      << "  --no-reference   omit the additional FP64 reference in GPU runs\n"
                      << "  --strict-nan     require identical NaN bits (default)\n"
                      << "  --nan-classification  ignore NaN payload/sign differences\n"
                      << "  --benchmark      benchmark available GPU paths (CUDA: integer emulation only)\n"
                      << "  --benchmark-only skip stress suites; still verify benchmark outputs\n"
                      << "  --bench-tiles N  benchmark tiles per launch (default 2048)\n"
                      << "  --bench-inner N  resident-mode recurrences (default 32)\n"
                      << "  --bench-repeats N  launches per timing sample (default 5)\n"
                      << "  --bench-samples N  timing samples per kernel (default 5)\n"
                      << "  --report FILE    write JSON results\n"
                      << "  --failure FILE   save first full failing tile for replay\n"
                      << "  --replay FILE    check a previously saved tile\n"
                      << "Representative special values, boundary sweeps, all input encodings,\n"
                      << "lane mapping, and golden cases always run (except in replay mode).\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + arg);
    }
    if (o.dtype != "bf16" && o.dtype != "f16" && o.dtype != "both") throw std::runtime_error("dtype must be bf16, f16, or both");
    if (o.cpu_only && !o.reference) throw std::runtime_error("CPU-only validation needs the independent reference");
    if (o.cpu_only && o.benchmark) throw std::runtime_error("GPU benchmarking requires a GPU build");
    if (o.benchmark && !o.replay.empty()) throw std::runtime_error("benchmark and replay are separate modes");
    return o;
}
struct BenchmarkResult {
    std::string dtype, mode, backend;
    int tiles, iterations;
    double median_us, min_us, max_us, effective_tflops, relative_to_wmma;
};
struct Result {
    std::string dtype, suite;
    uint64_t seed = 0, tiles = 0, outputs = 0, nan_outputs = 0, diagonal_cases = 0;
};

std::string json_string(const std::string& s) {
    std::ostringstream out; out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << char(c);
    }
    out << '"'; return out.str();
}
void save_failure(const std::string& path, Format format, const Tile& tile,
                  const std::string& suite, uint64_t index) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("cannot write failure file: " + path);
    file << "RDNA35_WMMA_TILE_V1 " << name(format) << '\n';
    for (auto x : tile.a) file << std::hex << std::setw(4) << std::setfill('0') << x << ' ';
    file << '\n'; for (auto x : tile.bt) file << std::hex << std::setw(4) << x << ' ';
    file << '\n'; for (auto x : tile.c) file << std::hex << std::setw(8) << x << ' ';
    file << "\n# suite=" << suite << " tile=" << std::dec << index << '\n';
    if (!file) throw std::runtime_error("failed writing failure tile");
}
std::pair<Format, Tile> read_failure(const std::string& path) {
    std::ifstream file(path); std::string magic, dtype;
    if (!(file >> magic >> dtype) || magic != "RDNA35_WMMA_TILE_V1" || (dtype != "bf16" && dtype != "f16"))
        throw std::runtime_error("invalid replay header");
    Tile tile; uint64_t value;
    for (auto* array : {&tile.a, &tile.bt}) for (auto& x : *array) {
        if (!(file >> std::hex >> value) || value > 65535) throw std::runtime_error("invalid replay input"); x = uint16_t(value);
    }
    for (auto& x : tile.c) { if (!(file >> std::hex >> value) || value > 0xffffffffu) throw std::runtime_error("invalid replay accumulator"); x = uint32_t(value); }
    return {dtype == "bf16" ? Format::BF16 : Format::FP16, tile};
}
#ifndef WMMA_CPU_ONLY
void hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) { hip_check(hipMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)), "hipMalloc"); }
    ~DeviceBuffer() { if (data) (void)hipFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
#endif

class Validator {
    Options opt;
    std::vector<Result> results;
    std::vector<BenchmarkResult> benchmarks;
    std::string device = "CPU only", arch = "none";
    uint64_t golden_checks = 0;
#ifndef WMMA_CPU_ONLY
    std::unique_ptr<DeviceBuffer<uint16_t>> da, db;
    std::unique_ptr<DeviceBuffer<uint32_t>> dc, simulated, hardware, dot_outputs, dot_wave;
    hipStream_t stream = nullptr;
#endif
public:
    explicit Validator(const Options& o) : opt(o) {
#ifndef WMMA_CPU_ONLY
        if (!opt.cpu_only) {
            int count = 0; hip_check(hipGetDeviceCount(&count), "hipGetDeviceCount (use --cpu-only without a GPU)");
            if (!count) throw std::runtime_error("no GPU; use --cpu-only");
            hipDeviceProp_t prop{}; hip_check(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
            device = prop.name;
#if WMMA_CUDA
            arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);
#else
            arch = prop.gcnArchName;
            if (arch.rfind("gfx110", 0) != 0 && arch.rfind("gfx115", 0) != 0)
                throw std::runtime_error("hardware WMMA validation requires RDNA3/3.5, got " + arch);
#endif
            hip_check(hipSetDevice(0), "hipSetDevice");
            da = std::make_unique<DeviceBuffer<uint16_t>>(size_t(opt.batch) * 256);
            db = std::make_unique<DeviceBuffer<uint16_t>>(size_t(opt.batch) * 256);
            dc = std::make_unique<DeviceBuffer<uint32_t>>(size_t(opt.batch) * 256);
            simulated = std::make_unique<DeviceBuffer<uint32_t>>(size_t(opt.batch) * 256);
#if WMMA_AMD_GPU
            hardware = std::make_unique<DeviceBuffer<uint32_t>>(size_t(opt.batch) * 256);
            dot_outputs = std::make_unique<DeviceBuffer<uint32_t>>(size_t(opt.batch) * 256);
            dot_wave = std::make_unique<DeviceBuffer<uint32_t>>(size_t(opt.batch) * 256);
#endif
            hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreate");
        }
#endif
        std::cout << "Device: " << device << " (" << arch << ")\n"
                  << "NaNs: " << (opt.strict_nan ? "strict raw bits" : "classification only; payload/sign excluded") << '\n';
    }
    ~Validator() {
#ifndef WMMA_CPU_ONLY
        if (stream) (void)hipStreamDestroy(stream);
#endif
    }
    void compare(Format format, Result& result, const std::vector<Tile>& tiles) {
        const size_t count = tiles.size(), elements = count * 256;
        std::vector<uint16_t> a(elements), b(elements);
        std::vector<uint32_t> c(elements), cpu(elements), reference(elements), gpu(elements), hw(elements), ndot(elements), wave_dot(elements);
        for (size_t i = 0; i < count; ++i) {
            std::copy(tiles[i].a.begin(), tiles[i].a.end(), a.begin() + i * 256);
            std::copy(tiles[i].bt.begin(), tiles[i].bt.end(), b.begin() + i * 256);
            std::copy(tiles[i].c.begin(), tiles[i].c.end(), c.begin() + i * 256);
        }
        auto work = [&](int worker) {
            for (size_t i = worker; i < count; i += opt.threads) {
                const size_t base = i * 256;
                if (format == Format::BF16) simulate_wmma_bf16_cpu(a.data() + base, b.data() + base, c.data() + base, cpu.data() + base);
                else simulate_wmma_f16_cpu(a.data() + base, b.data() + base, c.data() + base, cpu.data() + base);
                if (opt.reference) for (int cell = 0; cell < 256; ++cell) {
                    const uint16_t* av = a.data() + base + (cell / 16) * 16;
                    const uint16_t* bv = b.data() + base + (cell % 16) * 16;
                    reference[base + cell] = format == Format::BF16 ? reference_output<Format::BF16>(av, bv, c[base + cell])
                                                                   : reference_output<Format::FP16>(av, bv, c[base + cell]);
                }
            }
        };
        std::vector<std::thread> workers;
        for (int i = 1; i < opt.threads; ++i) workers.emplace_back(work, i);
        work(0); for (auto& thread : workers) thread.join();
#ifndef WMMA_CPU_ONLY
        if (!opt.cpu_only) {
            hip_check(hipMemcpyAsync(da->data, a.data(), elements * 2, hipMemcpyHostToDevice, stream), "copy A");
            hip_check(hipMemcpyAsync(db->data, b.data(), elements * 2, hipMemcpyHostToDevice, stream), "copy B");
            hip_check(hipMemcpyAsync(dc->data, c.data(), elements * 4, hipMemcpyHostToDevice, stream), "copy C");
            if (format == Format::BF16) {
                simulate_wmma_bf16_gpu<<<(elements + 255) / 256, 256, 0, stream>>>(da->data, db->data, dc->data, simulated->data, int(count));
                hip_check(hipGetLastError(), "launch BF16 software emulator");
#if WMMA_AMD_GPU
                hardware_wmma<Format::BF16><<<count, 32, 0, stream>>>(da->data, db->data, dc->data, hardware->data);
#endif
            } else {
                simulate_wmma_f16_gpu<<<(elements + 255) / 256, 256, 0, stream>>>(da->data, db->data, dc->data, simulated->data, int(count));
                hip_check(hipGetLastError(), "launch FP16 software emulator");
#if WMMA_AMD_GPU
                hardware_wmma<Format::FP16><<<count, 32, 0, stream>>>(da->data, db->data, dc->data, hardware->data);
#endif
            }
#if WMMA_AMD_GPU
            hip_check(hipGetLastError(), "launch hardware WMMA oracle");
            if (format == Format::BF16) {
                native_dot2_outputs<Format::BF16><<<(elements+255)/256,256,0,stream>>>(da->data,db->data,dc->data,dot_outputs->data,int(count),1);
                native_dot2_wave<Format::BF16><<<count,32,0,stream>>>(da->data,db->data,dc->data,dot_wave->data,1);
            } else {
                native_dot2_outputs<Format::FP16><<<(elements+255)/256,256,0,stream>>>(da->data,db->data,dc->data,dot_outputs->data,int(count),1);
                native_dot2_wave<Format::FP16><<<count,32,0,stream>>>(da->data,db->data,dc->data,dot_wave->data,1);
            }
            hip_check(hipGetLastError(), "launch native dot2 validation kernels");
            hip_check(hipMemcpyAsync(ndot.data(),dot_outputs->data,elements*4,hipMemcpyDeviceToHost,stream),"copy native dot2 outputs");
            hip_check(hipMemcpyAsync(wave_dot.data(),dot_wave->data,elements*4,hipMemcpyDeviceToHost,stream),"copy native dot2 wave");
#endif
            hip_check(hipMemcpyAsync(gpu.data(), simulated->data, elements * 4, hipMemcpyDeviceToHost, stream), "copy GPU emulation");
#if WMMA_AMD_GPU
            hip_check(hipMemcpyAsync(hw.data(), hardware->data, elements * 4, hipMemcpyDeviceToHost, stream), "copy hardware WMMA");
#endif
            hip_check(hipStreamSynchronize(stream), "finish GPU validation");
        }
#endif
        for (size_t i = 0; i < elements; ++i) {
            const bool bad_reference = opt.reference && !equal(cpu[i], reference[i], opt.strict_nan);
            bool bad_gpu = !opt.cpu_only && !equal(cpu[i], gpu[i], opt.strict_nan);
#if WMMA_AMD_GPU
            bad_gpu |= !opt.cpu_only && (!equal(cpu[i], hw[i], opt.strict_nan) || !equal(gpu[i], hw[i], opt.strict_nan) || !equal(ndot[i],hw[i],opt.strict_nan) || !equal(wave_dot[i],hw[i],opt.strict_nan));
#endif
            if (bad_reference || bad_gpu) {
                const uint64_t tile_index = result.tiles + i / 256;
                save_failure(opt.failure, format, tiles[i / 256], result.suite, tile_index);
                std::ostringstream msg;
                msg << name(format) << ' ' << result.suite << " tile=" << tile_index
                    << " row=" << (i % 256) / 16 << " col=" << i % 16 << std::hex
                    << " CPU=0x" << cpu[i] << " reference=0x" << reference[i]
                    << " GPU=0x" << gpu[i]
#if WMMA_AMD_GPU
                    << " native_dot=0x" << ndot[i] << " wave_dot=0x" << wave_dot[i] << " hardware=0x" << hw[i]
#endif
                    << std::dec
                    << "; replay tile saved to " << opt.failure;
                throw std::runtime_error("MISMATCH: " + msg.str());
            }
            result.nan_outputs += is_nan(cpu[i]);
        }
        result.tiles += count; result.outputs += elements;
    }
    void finish(Result result) {
        results.push_back(result);
        std::cout << name(result.dtype == "bf16" ? Format::BF16 : Format::FP16) << ' ' << result.suite
                  << ": " << result.tiles << " tiles, " << result.outputs << " outputs, "
                  << result.nan_outputs << " NaNs, 0 mismatches\n" << std::flush;
    }
    void golden_cases() {
        for (Format f : {Format::BF16, Format::FP16}) {
            const uint16_t one = encode16(f, false, 0), neg_one = one ^ 0x8000;
            const uint16_t small = encode16(f, false, -12);
            auto check = [&](uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1, uint32_t c, uint32_t expected) {
                const uint32_t actual = f == Format::BF16 ? simulate_dot2<Format::BF16>(a0,b0,a1,b1,c)
                                                         : simulate_dot2<Format::FP16>(a0,b0,a1,b1,c);
                if (!equal(actual, expected, true)) throw std::runtime_error("CPU arithmetic golden case failed");
                ++golden_checks;
            };
            check(one,one,0,0,0xbf800000,as_bits(0x1p-23f));
            check(neg_one,one,0,0,0x3f800000,as_bits(-0x1p-24f));
            check(one,one,neg_one,one,0,as_bits(-0x1p-24f));
            check(one,one,small,small,0xbf800000,as_bits(0x1.8p-23f));
            check(0x8000,one,one,one,0,0x3f7fffff);
            check(0,0,0,0,0x80000001,0);
            check(0,0,0,0,0x80800000,0x80800000);
            check(0,0,0,0,0x80000000,0);
            const uint16_t inf = f == Format::BF16 ? 0x7f80 : 0x7c00;
            check(inf,0,0,0,0,INVALID_NAN);
            check(inf,0,0,0,0x7fc12345,INVALID_NAN);
            check(uint16_t(inf|1),one,inf,0,0xffc12345,OPERAND_NAN);
            check(inf,0,uint16_t(inf|1),one,0,OPERAND_NAN);
            check(0,0,0,0,0x7f800001,0x7fc00001);
            check(0,0,0,0,0xff812345,0xffc12345);
            check(inf,one,0,0,0x7f812345,0x7fc12345);
            check(inf,one,uint16_t(inf^0x8000),one,0x7fc12345,INVALID_NAN);
            if (f == Format::BF16) {
                check(encode16(f,false,-100),encode16(f,false,-51),0x8000,one,0x80800000,0);
                check(encode16(f,false,-100),encode16(f,false,-52),0x8000,one,0x80800000,0x80800000);
                check(0x8001,one,one,one,0,0x3f7fffff);
            } else check(0x8001,one,0,0,0,as_bits(-0x1p-24f - 0x1p-38f));
        }
        std::cout << "Golden dot2 cases: " << golden_checks << " passed\n";
    }
    template<class Generator>
    void suite(Format f, const std::string& label, uint64_t total, uint64_t seed, Generator generator, uint64_t cases = 0) {
        Result result; result.dtype = name(f); result.suite = label; result.seed = seed; result.diagonal_cases = cases;
        for (uint64_t offset = 0; offset < total;) {
            const size_t n = size_t(std::min<uint64_t>(opt.batch, total - offset));
            std::vector<Tile> batch; batch.reserve(n);
            for (size_t i = 0; i < n; ++i) batch.push_back(generator(offset + i));
            compare(f, result, batch); offset += n;
        }
        finish(result);
    }
    void mapping(Format f) {
        // One-hot B selects each k independently, for every output row/column.
        // The selected integer values are exactly representable in both formats.
        suite(f, "lane_mapping", 16, 0, [&](uint64_t k) {
            Tile t;
            for (int m = 0; m < 16; ++m) for (int j = 0; j < 16; ++j) t.a[m*16+j] = encode16(f,false,(m+j)%4);
            for (int n = 0; n < 16; ++n) t.bt[n*16+(k+n)%16] = encode16(f,false,0);
            for (int m = 0; m < 16; ++m) for (int n = 0; n < 16; ++n) {
                const uint32_t expected = encode32(false,(m+int((k+n)%16))%4);
                const uint32_t actual = f == Format::BF16 ? simulate_output<Format::BF16>(t.a.data()+m*16,t.bt.data()+n*16,0)
                                                         : simulate_output<Format::FP16>(t.a.data()+m*16,t.bt.data()+n*16,0);
                if (actual != expected) throw std::runtime_error("independent lane-mapping golden case failed");
                ++golden_checks;
            }
            return t;
        });
    }
    void special(Format f) {
        const uint16_t inf = f == Format::BF16 ? 0x7f80 : 0x7c00;
        const uint16_t max = inf - 1, tiny = f == Format::BF16 ? 0x0080 : 0x0400;
        const uint16_t one = encode16(f,false,0);
        const std::array<uint16_t,13> values = {0,0x8000,one,uint16_t(one^0x8000),inf,uint16_t(inf^0x8000),
            uint16_t(inf|1),max,uint16_t(max^0x8000),tiny,uint16_t(tiny^0x8000),1,0x8001};
        const std::array<uint32_t,17> cv = {0,0x80000000,0x3f800000,0xbf800000,INF,INF|0x80000000,QNAN,
            0x7f7fffff,0xff7fffff,0x00800000,0x80800000,1,0x80000001,0x007fffff,0x807fffff,0x00800001,0x80800001};
        const uint64_t cases = 13ull*13*13*13*17;
        suite(f,"special_values",(cases+15)/16,0,[&](uint64_t tile_index) {
            Tile t;
            for (int lane=0;lane<16;++lane) {
                uint64_t id=tile_index*16+lane; if(id>=cases) break;
                t.c[lane*16+lane]=cv[id%17]; id/=17;
                const auto b1=values[id%13]; id/=13;
                const auto a1=values[id%13]; id/=13;
                const auto b0=values[id%13]; id/=13;
                const auto a0=values[id%13];
                t.a[lane*16]=a0; t.a[lane*16+1]=a1; t.bt[lane*16]=b0; t.bt[lane*16+1]=b1;
            }
            return t;
        }, cases);
    }
    void boundary(Format f) {
        const int low=f==Format::BF16?-100:-24, high=f==Format::BF16?-50:-1;
        const uint64_t length=high-low+1, cases=length*length*8*17;
        suite(f,"normal_boundary",(cases+15)/16,0,[&](uint64_t tile_index) {
            Tile t;
            for(int lane=0;lane<16;++lane) {
                uint64_t id=tile_index*16+lane; if(id>=cases)break;
                const int offset=int(id%17)-8; id/=17;
                const bool cs=id%2; id/=2; const bool bs=id%2; id/=2; const bool as=id%2; id/=2;
                const int eb=low+int(id%length); id/=length; const int ea=low+int(id%length);
                // FP16 powers below -14 are subnormal encodings; BF16 powers
                // in this sweep are all normal.
                auto power=[&](int e,bool sign) {
                    if(f==Format::FP16 && e < -14) return uint16_t((sign?0x8000:0)|(1u<<(e+24)));
                    return encode16(f,sign,e);
                };
                t.a[lane*16]=power(ea,as); t.bt[lane*16]=power(eb,bs);
                t.a[lane*16+1]=cs?0x8000:0; t.bt[lane*16+1]=encode16(f,false,0);
                t.c[lane*16+lane]=uint32_t(0x00800000+offset)|(cs?0x80000000u:0);
            }
            return t;
        },cases);
    }
    void all_encodings(Format f) {
        // Every 16-bit input encoding appears in each multiplicand role, with
        // cancellation, signed zero, infinities, and varied accumulators.
        const uint16_t one=encode16(f,false,0);
        for(int role=0;role<4;++role) suite(f,"all_encodings_role_"+std::to_string(role),4096,0,[&](uint64_t tile_index) {
            Tile t;
            for(int lane=0;lane<16;++lane) {
                const uint16_t x=uint16_t(tile_index*16+lane);
                uint16_t v[4]={one,one,uint16_t(one^0x8000),one}; v[role]=x;
                t.a[lane*16]=v[0];t.bt[lane*16]=v[1];t.a[lane*16+1]=v[2];t.bt[lane*16+1]=v[3];
                const uint32_t cs[]={0,0x80000000,0x3f800000,0xbf800000,0x00800000,0x80800000,INF,QNAN};
                t.c[lane*16+lane]=cs[x%8];
            }
            return t;
        },65536);
    }
#ifndef WMMA_CPU_ONLY
    template<Format F>
    void launch_benchmark(int backend, const uint16_t* a, const uint16_t* b, const uint32_t* c,
                          uint32_t* d, int tiles, int iterations) {
#if WMMA_AMD_GPU
        if (backend == 0) benchmark_wmma<F><<<tiles,32,0,stream>>>(a,b,c,d,iterations);
        else if (backend == 1) native_dot2_outputs<F><<<tiles,256,0,stream>>>(a,b,c,d,tiles,iterations);
        else if (backend == 2) native_dot2_wave<F><<<tiles,32,0,stream>>>(a,b,c,d,iterations);
        else
#endif
            benchmark_integer<F><<<tiles,256,0,stream>>>(a,b,c,d,tiles,iterations);
        hip_check(hipGetLastError(), "launch benchmark kernel");
    }
    void benchmark_format(Format f) {
        const int tiles = opt.bench_tiles;
        const size_t elements = size_t(tiles)*256;
        std::vector<uint16_t> a(elements), b(elements);
        std::vector<uint32_t> c(elements,0), expected(elements), actual(elements);
        RNG rng{opt.seed + 991 + (f == Format::BF16 ? 0u : 100u)};
        for (size_t i=0;i<elements;++i) {
            const int ae=rng.range(-6,-3), be=rng.range(-6,-3);
            const auto am=uint32_t(rng.next()), bm=uint32_t(rng.next());
            a[i]=encode16(f,false,ae,am); b[i]=encode16(f,false,be,bm);
        }
        DeviceBuffer<uint16_t> ga(elements), gb(elements);
        DeviceBuffer<uint32_t> gc(elements), gd(elements);
        hip_check(hipMemcpyAsync(ga.data,a.data(),elements*2,hipMemcpyHostToDevice,stream),"benchmark A");
        hip_check(hipMemcpyAsync(gb.data,b.data(),elements*2,hipMemcpyHostToDevice,stream),"benchmark B");
        hip_check(hipMemcpyAsync(gc.data,c.data(),elements*4,hipMemcpyHostToDevice,stream),"benchmark C");
        hip_check(hipStreamSynchronize(stream),"benchmark data ready");
        auto launch = [&](int backend,int iterations) {
            if (f == Format::BF16) launch_benchmark<Format::BF16>(backend,ga.data,gb.data,gc.data,gd.data,tiles,iterations);
            else launch_benchmark<Format::FP16>(backend,ga.data,gb.data,gc.data,gd.data,tiles,iterations);
        };
#if WMMA_AMD_GPU
        const char* names[] = {"wmma", "native_dot2_per_output", "native_dot2_per_wave", "integer_emulation"};
#else
        const char* names[] = {"integer_emulation"};
#endif
        constexpr int backend_count = sizeof(names) / sizeof(names[0]);
        for (int mode=0;mode<2;++mode) {
            const int iterations = mode ? opt.bench_inner : 1;
            auto cpu_work = [&](int worker) {
                for (size_t i=worker;i<elements;i+=opt.threads) {
                    const size_t base=(i/256)*256,cell=i%256;
                    uint32_t value=c[i];
                    for(int repeat=0;repeat<iterations;++repeat)
                        value=f==Format::BF16 ? simulate_output<Format::BF16>(a.data()+base+(cell/16)*16,b.data()+base+(cell%16)*16,value)
                                              : simulate_output<Format::FP16>(a.data()+base+(cell/16)*16,b.data()+base+(cell%16)*16,value);
                    expected[i]=value;
                }
            };
            std::vector<std::thread> workers;
            for(int i=1;i<opt.threads;++i)workers.emplace_back(cpu_work,i);
            cpu_work(0);for(auto& worker:workers)worker.join();
            // Verify every available path before timing, on the same finite data.
            for(int backend=0;backend<backend_count;++backend) {
                launch(backend,iterations);
                hip_check(hipMemcpyAsync(actual.data(),gd.data,elements*4,hipMemcpyDeviceToHost,stream),"benchmark result");
                hip_check(hipStreamSynchronize(stream),"verify benchmark");
                for(size_t i=0;i<elements;++i)if(actual[i]!=expected[i])
                    throw std::runtime_error(std::string("benchmark correctness mismatch: ")+name(f)+" "+names[backend]+" iterations="+std::to_string(iterations)+" output="+std::to_string(i));
                for(int warm=0;warm<3;++warm)launch(backend,iterations);
            }
            hip_check(hipStreamSynchronize(stream),"finish benchmark warmup");
            std::array<std::vector<double>,backend_count> samples;
            hipEvent_t begin=nullptr,end=nullptr;
            hip_check(hipEventCreate(&begin),"benchmark event begin");
            try {
                hip_check(hipEventCreate(&end),"benchmark event end");
                for(int sample=0;sample<opt.bench_samples;++sample)for(int offset=0;offset<backend_count;++offset) {
                    const int backend=(sample+offset)%backend_count; // interleave to reduce order bias
                    hip_check(hipEventRecord(begin,stream),"benchmark begin");
                    for(int repeat=0;repeat<opt.bench_repeats;++repeat)launch(backend,iterations);
                    hip_check(hipEventRecord(end,stream),"benchmark end");
                    hip_check(hipEventSynchronize(end),"wait benchmark");
                    float ms=0;hip_check(hipEventElapsedTime(&ms,begin,end),"benchmark elapsed");
                    samples[backend].push_back(double(ms)*1000/opt.bench_repeats);
                }
            } catch(...) {if(end)(void)hipEventDestroy(end);(void)hipEventDestroy(begin);throw;}
            (void)hipEventDestroy(end);(void)hipEventDestroy(begin);
            for(auto& values:samples)std::sort(values.begin(),values.end());
            const double wmma_us=samples[0][samples[0].size()/2];
            if(wmma_us<=0)throw std::runtime_error("benchmark below HIP event timer resolution; increase repeats/tiles/inner");
            for(int backend=0;backend<backend_count;++backend) {
                const auto& values=samples[backend];const double median=values[values.size()/2];
                if(median<=0)throw std::runtime_error("zero benchmark time");
                BenchmarkResult row{name(f),mode?"register_resident":"tile_kernel",names[backend],tiles,iterations,
                                    median,values.front(),values.back(),8192.0*tiles*iterations/median/1e6,WMMA_AMD_GPU ? median/wmma_us : 0.0};
                benchmarks.push_back(row);
                std::cout<<"BENCH "<<row.dtype<<' '<<row.mode<<' '<<row.backend<<": "<<row.median_us
                         <<" us, "<<row.effective_tflops<<" effective TFLOP/s";
                if (WMMA_AMD_GPU) std::cout<<", "<<row.relative_to_wmma<<"x WMMA time";
                std::cout<<'\n'<<std::flush;
            }
        }
    }
#endif
    void run_benchmarks() {
#ifndef WMMA_CPU_ONLY
        for(Format f:{Format::BF16,Format::FP16})if(opt.dtype=="both"||opt.dtype==name(f))benchmark_format(f);
#else
        throw std::runtime_error("benchmark needs a GPU build");
#endif
    }
    void run() {
        golden_cases();
        if(opt.benchmark_only) { run_benchmarks(); return; }
        if(!opt.replay.empty()) {
            auto saved=read_failure(opt.replay);
            Result result;result.dtype=name(saved.first);result.suite="replay";
            compare(saved.first,result,{saved.second});finish(result);return;
        }
        const std::array<std::string,4> labels={"bounded","bits","cancellation","exponents"};
        for(Format f:{Format::BF16,Format::FP16}) {
            if(opt.dtype!="both" && opt.dtype!=name(f)) continue;
            mapping(f);
            for(size_t i=0;i<labels.size();++i) {
                const uint64_t seed=opt.seed+(f==Format::BF16?0:100)+i;
                RNG rng{seed};
                suite(f,labels[i],opt.trials,seed,[&](uint64_t){return random_tile(f,labels[i],rng);});
            }
            special(f);boundary(f);all_encodings(f);
        }
        if(opt.benchmark) run_benchmarks();
    }
    void report(double seconds, bool passed, const std::string& error) {
        uint64_t outputs=0,nans=0;for(const auto& r:results){outputs+=r.outputs;nans+=r.nan_outputs;}
        const auto now = std::time(nullptr);
        std::ostringstream timestamp; timestamp << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%SZ");
        std::ostringstream j;
        j << "{\n  \"passed\": " << (passed?"true":"false")
          << ",\n  \"timestamp_utc\": " << json_string(timestamp.str())
          << ",\n  \"source_sha256\": " << json_string(WMMA_SOURCE_SHA256)
          << ",\n  \"compiler\": " << json_string(__VERSION__)
          << ",\n  \"device\": " << json_string(device) << ",\n  \"architecture\": " << json_string(arch)
          << ",\n  \"hardware_verified\": " << (WMMA_AMD_GPU && !opt.cpu_only && passed && (outputs || !benchmarks.empty()) ? "true" : "false")
          << ",\n  \"fp64_reference_enabled\": " << (opt.reference && !opt.benchmark_only ? "true":"false")
          << ",\n  \"strict_nan_bits\": " << (opt.strict_nan?"true":"false")
          << ",\n  \"seed\": " << opt.seed << ",\n  \"trials_per_random_family\": " << opt.trials
          << ",\n  \"batch\": " << opt.batch << ",\n  \"threads\": " << opt.threads
          << ",\n  \"benchmark_only\": " << (opt.benchmark_only?"true":"false")
          << ",\n  \"bench_repeats\": " << opt.bench_repeats << ",\n  \"bench_samples\": " << opt.bench_samples
          << ",\n  \"golden_checks\": " << golden_checks << ",\n  \"outputs\": " << outputs
          << ",\n  \"non_nan_outputs\": " << outputs-nans << ",\n  \"nan_classifications\": " << nans
          << ",\n  \"bitwise_checked_output_positions\": " << (opt.strict_nan ? outputs : outputs-nans)
          << ",\n  \"nan_bitwise_outputs\": " << (opt.strict_nan ? nans : 0)
          << ",\n  \"benchmark_checked_output_positions\": " << uint64_t(benchmarks.size())*uint64_t(opt.bench_tiles)*256
          << ",\n  \"elapsed_seconds\": " << seconds << ",\n  \"error\": " << json_string(error) << ",\n  \"results\": [";
        for(size_t i=0;i<results.size();++i){const auto&r=results[i];
            j << (i?",\n    ":"\n    ") << "{\"dtype\":" << json_string(r.dtype) << ",\"suite\":" << json_string(r.suite)
              << ",\"seed\":" << r.seed << ",\"tiles\":" << r.tiles << ",\"outputs\":" << r.outputs
              << ",\"nan_outputs\":" << r.nan_outputs << ",\"diagonal_cases\":" << r.diagonal_cases << ",\"mismatches\":0}";}
        j << "\n  ],\n  \"benchmarks\": [";
        for(size_t i=0;i<benchmarks.size();++i){const auto&r=benchmarks[i];
            j<<(i?",\n    ":"\n    ")<<"{\"dtype\":"<<json_string(r.dtype)<<",\"mode\":"<<json_string(r.mode)
             <<",\"backend\":"<<json_string(r.backend)<<",\"tiles\":"<<r.tiles<<",\"iterations\":"<<r.iterations
             <<",\"median_us\":"<<r.median_us<<",\"min_us\":"<<r.min_us<<",\"max_us\":"<<r.max_us
             <<",\"effective_tflops\":"<<r.effective_tflops<<",\"relative_to_wmma\":";
            if (WMMA_AMD_GPU) j<<r.relative_to_wmma; else j<<"null";
            j<<"}";}
        j << "\n  ]\n}\n";
        if(!opt.report.empty()) {std::ofstream out(opt.report);if(!out)throw std::runtime_error("cannot write report: "+opt.report);out<<j.str();if(!out)throw std::runtime_error("failed writing report");}
        if(!results.empty() || benchmarks.empty()) std::cout << (passed?"PASS":"FAIL") << ": " << outputs << " output positions; " << outputs-nans
                  << " non-NaN bitwise checks; " << nans << " NaN " << (opt.strict_nan?"bitwise checks":"classifications")
                  << "; " << seconds << " seconds.\n";
        if(opt.cpu_only) std::cout << "CPU integer vs FP64 reference only; hardware equivalence was NOT checked.\n";
        else if (WMMA_CUDA) std::cout << "CUDA GPU integer emulation checked against the CPU model; RDNA hardware was NOT checked.\n";
        else std::cout << "CPU/GPU integer emulation and both native dot2 kernels checked against real WMMA.\n";
        if(!benchmarks.empty())std::cout<<(passed?"PASS benchmark: ":"FAIL benchmark: ")<<benchmarks.size()<<" rows, all timed paths correctness-checked; transfers and CPU work excluded.\n";
    }
};
} // namespace rdna35

int main(int argc,char**argv) {
    try {
        const auto opt=rdna35::options(argc,argv);
        if(std::fesetround(FE_TONEAREST)!=0)throw std::runtime_error("cannot set host reference rounding to nearest-even");
        rdna35::Validator validator(opt);
        const auto start=std::chrono::steady_clock::now();
        try {validator.run();}
        catch(const std::exception& e) {
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            validator.report(seconds,false,e.what());std::cerr<<e.what()<<'\n';return 1;
        }
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        validator.report(seconds,true,"");return 0;
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
