#pragma once

#include <immintrin.h>

/*
 * Fastest way to horizontally add floats of a 256-bit register.
 * Courtesy of: https://stackoverflow.com/a/35270026/6920681
 */
float hsum256_ps_avx(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    vlow = _mm_add_ps(vlow, vhigh);     // add the low 128
    __m128 shuf = _mm_movehdup_ps(vlow);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

template<typename It, typename T, typename Compare = std::less<>>
auto lower_bound_branchless(It low, It last, const T& val, Compare lt = {})
{
    auto n = std::distance(low, last);

    while (auto half = n / 2)
    {
        auto middle = low;
        std::advance(middle, half);
        low = lt(*middle, val) ? middle : low;
        n -= half;
    }
    if (lt(*low, val))
        ++low;
    return low;
}