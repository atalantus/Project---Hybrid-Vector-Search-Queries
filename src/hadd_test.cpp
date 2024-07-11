#include <iostream>
#include <cstdint>
#include <vector>
#include <immintrin.h>
#include <iomanip>

uint64_t m1 = 0;
uint64_t m2 = 0;

float hsum256_ps_avx(__m256 v)
{
    uint64_t s = __rdtsc();
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    vlow = _mm_add_ps(vlow, vhigh);     // add the low 128
    __m128 shuf = _mm_movehdup_ps(vlow);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    float f = _mm_cvtss_f32(sums);
    uint64_t e = __rdtsc();
    m1 += e - s;
    return f;
}

float mm256_hadd_ps(__m256 x)
{
    uint64_t s = __rdtsc();
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    float f = _mm_cvtss_f32(sum);

    uint64_t e = __rdtsc();
    m2 += e - s;
    return f;
}

int main()
{
    for (int i = 0; i < 100; i++)
    {
        __m256 v = _mm256_set_ps(i * 0.01232, i * (-23.4824), i * 55543.4343, i * 1.2323,
                                 i * 2.1234234, i * (-234.324234), i * 0.234343, i * (-0.0012323));

        std::cout << mm256_hadd_ps(v) << " " << hsum256_ps_avx(v) << std::endl;
    }

    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;
    std::cout << m1 / 10.0 << std::endl;
    std::cout << m2 / 10.0 << std::endl;
}