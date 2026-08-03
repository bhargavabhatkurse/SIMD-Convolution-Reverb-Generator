// convolution.cpp
//
// Speeding up a convolution reverb kernel one step at a time with SIMD.
// Convolving an audio signal with a room impulse response is basically a reverb
// effect. The inner loop is just a long multiply-accumulate (a dot product),
// which is a really clean thing to try and speed up with SIMD.
//
// I build it up in five rungs, and each rung fixes ONE specific bottleneck in
// the rung before it.
//
//   1. scalar baseline       - plain C++. This is the reference AND the number
//                              every other rung is measured against.
//   2. auto-vectorized       - same loop, but I let the compiler use SIMD.
//   3. AVX2 + FMA            - hand-written, 8 floats wide, one accumulator.
//   4. multiple accumulators - hide the FMA latency by running several at once.
//   5. register blocking     - reuse each tap across many outputs.
//
// Everything here is SINGLE-THREADED on purpose. I wanted to see how far raw
// SIMD and instruction-level parallelism get me on one core before threads
// enter the picture, so nothing below hides behind extra cores.
//
// Build and run:  make run
//
// Build flags note: the whole file compiles at one optimization level
// (-O3 -march=x86-64-v3, set in the Makefile). That matters because the scalar
// baseline is the denominator for every speedup, so it has to be built exactly
// the same way as the kernels it's compared against. The only per-function
// tweaks are:
//   - Rung 2 turns on fast-math, so the compiler is allowed to reorder the
//     float adds and actually vectorize the sum.
//
//   - Rungs 3-5 pin the avx2,fma target. That's redundant given -march, but it
//     keeps the intrinsics compiling even if someone builds with a weaker arch.

//-march=x86-64-v3:- assume the CPU has AVX2 and FMA available

//convolution: y[n] = Σ (k=0 to M-1) x[n - k] · h[k]

#include <immintrin.h>   // AVX2 / FMA intrinsics

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// ---- Rung 1: scalar baseline ------------------------------------------------
// y[n] = sum over k of hf[k] * xpad[n+k].
//
// Two bits of setup keep this loop clean. hf is the impulse response reversed
// (hf[k] = h[m-1-k]), and xpad is the input zero-padded by (m-1) on each side.
// Reversing turns a backward stride into forward, contiguous loads, and the
// padding means I never have to bounds-check inside the hot loop. Both of those
// pay off a lot once I get to SIMD.

void convolve_scalar(const std::vector<float>& xpad,
                     const std::vector<float>& hf,
                     std::vector<float>& y) {
    const size_t m = hf.size(), n_out = y.size();
    for (size_t n = 0; n < n_out; ++n) {
        float acc = 0.0f;
        for (size_t k = 0; k < m; ++k) //[0,m-1]
            acc += hf[k] * xpad[n + k];
        y[n] = acc;
    }
}

// ---- Rung 2: auto-vectorized ------------------------------------------------
// This is the EXACT same loop as Rung 1. The only difference is fast-math.
//
// On its own the compiler won't vectorize a float sum, because floating-point
// addition isn't associative, and adding the numbers in a different order can
// change the rounding. fast-math allows it, so now it's
// free to add 8 lanes at a time and fold in FMAs. This rung is here to show how
// much I get essentially for free before writing a single intrinsic.
#pragma GCC push_options
#pragma GCC optimize("fast-math")
void convolve_autovec(const std::vector<float>& xpad,
                      const std::vector<float>& hf,
                      std::vector<float>& y) {

    const size_t m = hf.size(), n_out = y.size();
    for (size_t n = 0; n < n_out; ++n) {
        float acc = 0.0f;
        for (size_t k = 0; k < m; ++k)
            acc += hf[k] * xpad[n + k];
        y[n] = acc;
    }
}

#pragma GCC pop_options

// ---- AVX2 kernels (Rungs 3-5) -----------------------------------------------
// Pin the avx2,fma target so every intrinsic below compiles no matter what
// -march the file is built with.
#pragma GCC push_options
#pragma GCC target("avx2,fma") //target machine supports these 

// Add up the 8 floats packed inside one vector register and return the total.
// This is the standard "horizontal sum" trick: fold the 256-bit register down
// to 128 bits, then keep halving until only one lane is left. 

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);          // low 4 floats
    __m128 hi = _mm256_extractf128_ps(v, 1);        // high 4 floats
    __m128 s  = _mm_add_ps(lo, hi);                 // now 4 partial sums
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));         // add upper 2 into lower 2
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x1));   // add the last pair
    return _mm_cvtss_f32(s); //lane-0 sum is returned as 1 single float
}

// Dot product for a single output, done with A separate accumulators.
// I made the accumulator count a template parameter so I can benchmark A = 2,
// 4, 8, 16 from one implementation instead of maintaining five near-identical
// copies. Rungs 4 calls this. Rung 3 writes the A = 1 case out by hand so the
// naive "before" version sits right next to it and is easy to compare.
//
// The loop runs in three phases:
//   1. main loop: chew through 8*A taps per iteration, A accumulators in flight
//   2. cleanup:   mop up remaining full groups of 8 with a single accumulator
//   3. tail:      the last few taps (< 8) done scalar
template <int A>
static inline float dot_avx2(const float* hp, const float* xw, size_t m) {
    __m256 acc[A]; //A 256 bit accumulators
    for (int a = 0; a < A; ++a) acc[a] = _mm256_setzero_ps(); //set all to 0

    size_t k = 0;
    const size_t step = 8 * A;
    for (; k + step <= m; k += step)                        // phase 1
        for (int a = 0; a < A; ++a)
        //fused multiply add - FMA
            acc[a] = _mm256_fmadd_ps(_mm256_loadu_ps(hp + k + 8 * a),
                                     _mm256_loadu_ps(xw + k + 8 * a), acc[a]);
            for (; k + 8 <= m; k += 8)                              // phase 2
                acc[0] = _mm256_fmadd_ps(_mm256_loadu_ps(hp + k),
                                    _mm256_loadu_ps(xw + k), acc[0]);

    __m256 sum = acc[0];                                    // combine the lanes
    for (int a = 1; a < A; ++a) sum = _mm256_add_ps(sum, acc[a]);
    float total = hsum256(sum); //collapse it into 1 float

    for (; k < m; ++k) total += hp[k] * xw[k];              // phase 3 (do sequntial MAC here)
    return total;
}

// Rung 3: hand-written AVX2, single accumulator.
// This is the honest "first SIMD attempt". It's correct and 8-wide, but every
// FMA has to wait on the result of the previous one (an FMA is ~4 cycles of
// latency), so the whole chain runs one FMA at a time. It's latency-bound: the
// hardware could be doing much more, but I've serialized it.
void convolve_avx2(const std::vector<float>& xpad, const std::vector<float>& hf,
                   std::vector<float>& y) {
    //8 floats fit on 1 avx register (256 bits) (single accumulator)

    const size_t m = hf.size(), n_out = y.size();
    
    const float* xp = xpad.data(); //.data() returns raw pointer to the vector's underlying array
    const float* hp = hf.data();

    for (size_t n = 0; n < n_out; ++n) {
        __m256 acc = _mm256_setzero_ps(); //zero initially
        
        size_t k = 0;
        for (; k + 8 <= m; k += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(hp + k),
                                  _mm256_loadu_ps(xp + n + k), acc);
        
        float total = hsum256(acc);
        
        for (; k < m; ++k) total += hp[k] * xp[n + k]; //remaining taps which were less than 8(hence couldnt be SIMD)
        y[n] = total;
    }
}

// Rung 4: A independent accumulators at once.
// The fix for Rung 3 is to have several accumulators that don't depend on each
// other. While FMA #1 is still in flight, FMA #2 into a different accumulator
// can already start. With enough of them, both of the CPU's FMA ports stay
// busy and I move from latency-bound toward throughput-bound. (This is just
// Little's Law: to fill a ~4-cycle-latency, 2-per-cycle pipeline I need roughly
// 8 operations in flight.)
template <int A>
void convolve_avx2_multi(const std::vector<float>& xpad,
                         const std::vector<float>& hf, std::vector<float>& y) {
    const size_t m = hf.size(), n_out = y.size();
    const float* xp = xpad.data();
    const float* hp = hf.data();
    for (size_t n = 0; n < n_out; ++n) y[n] = dot_avx2<A>(hp, xp + n, m);
}

// ---- Rung 5: register blocking, vectorized over OUTPUTS ---------------------
// Rungs 3 and 4 eventually hit the same wall. Every FMA needs two loads (one
// tap, one sample), and this CPU only has two load ports, so throughput tops
// out around 1 FMA per cycle -- about half of what the FMA units can do. Adding
// more accumulators doesn't help past that point; the loads are the limit.
//
// The trick is to change WHAT sits in the vector lanes. Instead of packing 8
// taps of one output, I pack 8 consecutive OUTPUTS into a register. Then for
// each tap I broadcast hf[k] once and reuse it across R output blocks. So per
// tap that's 1 broadcast + R sample loads feeding R*8 FMAs. As R grows, the
// loads-per-FMA ratio drops from 2 toward 1, which is how I get past the load
// port limit. Nice bonus: the answers land directly in the lanes, so there's
// no horizontal sum at the end at all.
//
//   y[n+j] = sum_k hf[k] * xpad[n+j+k]    (lane j of block r is output n+8r+j)
//
// R can't grow forever: R accumulators plus a temporary or two have to fit in
// the 16 YMM registers, so R up to about 8 works before the compiler spills to
// memory and it gets slower again.

template <int R>
static inline void compute_block(const float* hp, const float* xp, size_t m,
                                 size_t n, float* y) {
    __m256 acc[R];
    //At any point during this code, acc[r]'s lane j holds the running partial sum of output y[n + r*8 + j]
    //  accumulated over however many taps have been processed so far.
    for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();

    for (size_t k = 0; k < m; ++k) { //processing all taps
        __m256 hb = _mm256_broadcast_ss(hp + k);   // load tap once, reuse across blocks
        for (int r = 0; r < R; ++r)
            acc[r] = _mm256_fmadd_ps(hb, _mm256_loadu_ps(xp + n + r * 8 + k),
                                     acc[r]);
    }

    //at the end, each R will contain 1 8-output chunk;
    for (int r = 0; r < R; ++r) _mm256_storeu_ps(y + n + r * 8, acc[r]);
}

template <int R>
void convolve_avx2_block(const std::vector<float>& xpad,
                         const std::vector<float>& hf, std::vector<float>& y) {
    const size_t m = hf.size(), n_out = y.size();
    const float* xp = xpad.data();
    const float* hp = hf.data();

    const size_t width = 8 * R, nb = n_out / width;

    for (size_t b = 0; b < nb; ++b)
        compute_block<R>(hp, xp, m, b * width, y.data()); //start of the block: b * width
        
    for (size_t n = nb * width; n < n_out; ++n)   // outputs that don't fill a block
        y[n] = dot_avx2<4>(hp, xp + n, m); //rung-4
}

#pragma GCC pop_options


// ---- Setup helpers ----------------------------------------------------------
// One-time setup. None of this runs inside the timed region, so it doesn't
// affect the benchmark numbers.


std::vector<float> flip(const std::vector<float>& h) {
    return std::vector<float>(h.rbegin(), h.rend());
}

//pad the input signal
std::vector<float> zero_pad(const std::vector<float>& x, size_t m) {
    const size_t pad = m - 1;
    std::vector<float> xpad(x.size() + 2 * pad, 0.0f);
    std::copy(x.begin(), x.end(), xpad.begin() + pad);
    return xpad;
}

// Plain double-precision convolution. It's slow, but it's the ground truth I
// check every fast version against, so I trust it over the float kernels.

//(not Reversed Kernel) - Normal Convolution : y[n] = Σ (k=0 to M-1) x[n - k] · h[k]

std::vector<double> reference(const std::vector<float>& x,
                              const std::vector<float>& h) {
    std::vector<double> y(x.size() + h.size() - 1, 0.0);
    for (size_t n = 0; n < x.size(); ++n)
        for (size_t k = 0; k < h.size(); ++k)
            y[n + k] += double(x[n]) * double(h[k]);
    return y;
}

// ---- Timing and error-checking helpers --------------------------------------
//function pointer
using ConvFn = void (*)(const std::vector<float>&, const std::vector<float>&,
                        std::vector<float>&);

// Run a kernel a few times and take the median. The median throws out the odd
// slow run when the OS scheduler steals the core for a moment. The very first
// call is a warm-up (caches, page faults) and isn't timed.
double bench(ConvFn fn, const std::vector<float>& xpad,
             const std::vector<float>& hf, std::vector<float>& y, int runs) {
    
    fn(xpad, hf, y);  // warm-up, not counted

    std::vector<double> ms;
    for (int r = 0; r < runs; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        fn(xpad, hf, y); //call the implementation
        auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::sort(ms.begin(), ms.end());
    return ms[runs / 2]; //median
}

// Biggest absolute difference between a result and the fp64 reference. Accurate (close to the real answer)
double max_error(const std::vector<float>& y, const std::vector<double>& ref) {
    double e = 0.0;
    for (size_t i = 0; i < y.size(); ++i)
        e = std::max(e, std::fabs(double(y[i]) - ref[i]));
    return e;
}

// Biggest absolute difference between two float results. (checks: all your versions match))
double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::fabs(double(a[i]) - double(b[i])));
    return e;
}

// -----------------------------------------------------------------------------
int main() {
    struct Impl { const char* name; ConvFn fn; };

    const Impl impls[] = {
        {"scalar   (Rung 1)", convolve_scalar},          // 0
        {"autovec  (Rung 2)", convolve_autovec},         // 1
        {"avx2 x1  (Rung 3)", convolve_avx2},            // 2
        {"avx2 x2  (Rung 4)", convolve_avx2_multi<2>},   // 3
        {"avx2 x4  (Rung 4)", convolve_avx2_multi<4>},   // 4
        {"avx2 x8  (Rung 4)", convolve_avx2_multi<8>},   // 5
        {"avx2 x16 (Rung 4)", convolve_avx2_multi<16>},  // 6
        {"block r2 (Rung 5)", convolve_avx2_block<2>},   // 7
        {"block r4 (Rung 5)", convolve_avx2_block<4>},   // 8
        {"block r8 (Rung 5)", convolve_avx2_block<8>},   // 9
    };
    
    const int NI = int(sizeof(impls) / sizeof(impls[0])); //number of implementations

    // 1) Tiny example I can check by hand. Every version should print the same.
    {
        std::vector<float> x = {1, 2, 3}, h = {1, 1};
        std::vector<float> xp = zero_pad(x, h.size()), hff = flip(h);
        std::printf("sanity check  [1,2,3]*[1,1]  (expected 1 3 5 3)\n");
        
        for (const auto& im : impls) {
            std::vector<float> yy(x.size() + h.size() - 1);
            im.fn(xp, hff, yy);
            std::printf("  %-18s = [ ", im.name);
            for (float v : yy) std::printf("%.0f ", v);
            std::printf("]\n");
        }

        std::printf("\n");
    }

    // 2) Realistic sizes: ~1s of audio at 48 kHz, ~0.1s impulse response. (assuming fixed sample rate; 48 kHz)
    //samples = seconds × sample_rate
    const size_t N = 48000, M = 4800;
    const int    runs = 11;  //I will take the median of these


    std::mt19937 rng(1);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> x(N), h(M);
    for (auto& v : x) v = d(rng);
    for (auto& v : h) v = d(rng);

    std::vector<float> hf   = flip(h);
    std::vector<float> xpad = zero_pad(x, M);
    std::vector<float> y(N + M - 1);

    std::vector<double> ref = reference(x, h); //double precision convolution

    // Two flops per tap (one multiply, one add), over every output.
    auto gflops = [&](double ms) {
        return 2.0 * double(N + M - 1) * double(M) / (ms * 1e-3) / 1e9; //{EVERY MAC is 2 operations (multiply + add) })
    };

    // 3) Time every version. Keep each output so I can cross-check them below.
    std::printf("benchmark   N=%zu  M=%zu  (median of %d runs, single-threaded)\n\n",
                N, M, runs);
    std::printf("%-18s %10s %10s %9s %12s\n", "impl", "time(ms)", "GFLOP/s",
                "speedup", "err_vs_fp64");
    std::printf("--------------------------------------------------------------------\n");

    std::vector<std::vector<float>> out(NI);
    std::vector<double> tms(NI);
    double base_ms = 0.0;
    for (int i = 0; i < NI; ++i) {
        tms[i] = bench(impls[i].fn, xpad, hf, y, runs); //run each rung
        out[i] = y;
        double e = max_error(y, ref);

        if (i == 0) base_ms = tms[i]; //scalar baseline

        std::printf("%-18s %10.2f %10.2f %8.2fx %12.2e\n", impls[i].name,
                    tms[i], gflops(tms[i]), base_ms / tms[i], e);
    }

    // 4) Correctness: every version should match the scalar baseline, up to
    //    plain floating-point rounding.
    const double TOL = 1e-3;
    std::printf("\ncorrectness -- each rung vs the scalar baseline (Rung 1):\n");
    bool all_ok = true;
    for (int i = 0; i < NI; ++i) {
        double dd = max_diff(out[i], out[0]);
        bool ok = (dd <= TOL);
        all_ok = all_ok && ok;
        std::printf("  %-18s  max|d| vs scalar = %.2e   [%s]\n", impls[i].name,
                    dd, ok ? "OK" : "FAIL");
    }

    std::printf("  => %s  (tolerance %.0e; differences are pure float rounding)\n",
                all_ok ? "ALL RUNGS AGREE" : "MISMATCH!", TOL);


    // 5) Report the fastest kernel.
    int gbest = 0;
    for (int i = 1; i < NI; ++i) if (tms[i] < tms[gbest]) gbest = i;
    std::printf("\noverall best: %s = %.2fx over the scalar baseline "
                "(single core).\n", impls[gbest].name, base_ms / tms[gbest]);

    return 0;
}

