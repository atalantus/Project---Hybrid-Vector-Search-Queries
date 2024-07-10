#include <iostream>
#include <vector>
#include <immintrin.h>
#include <iomanip>

template<typename T>
T compare_with_id(const std::vector<T>& a, const std::vector<T>& b)
{
    size_t next = 10;

    T sum = 0.0;
    // Skip the first 2 dimensions
    for (size_t i = 2; i < a.size(); ++i)
    {
        T diff = a[i] - b[i];
        sum += diff * diff;

        if (i + 1 == next)
        {
            next += 8;
            std::cout << "seq after " << i + 1 << " steps, sum: " << std::setprecision(15) << sum << std::endl;
        }
    }

    std::cout << "seq final sum: " << std::setprecision(15) << sum << std::endl;

    return sum;
}

float mm256_hadd_ps(__m256 x)
{
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
    return _mm_cvtss_f32(sum);
}

float dist_to_query(const std::vector<float>& data_vec, const std::vector<float>& query_vec)
{
    __m256 sum_vec = _mm256_set1_ps(0.0);

    // Skip the first 2 dimensions
    for (size_t i = 2; i < 98; i += 8)
    {
        __m256 d_vec = _mm256_loadu_ps(&data_vec[i]);
        __m256 q_vec = _mm256_loadu_ps(&query_vec[i]);

        __m256 diff_vec = d_vec - q_vec;
        diff_vec *= diff_vec;
        sum_vec += diff_vec;

        std::cout << "simd after " << i + 8 << " steps, sum: " << std::setprecision(15) << mm256_hadd_ps(sum_vec)
                  << std::endl;
    }

    // do the rest
    {
        __m256i mask = _mm256_set_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
        __m256 d_vec = _mm256_castsi256_ps(
                _mm256_and_si256(_mm256_castps_si256(_mm256_loadu_ps(&data_vec[94])), mask));
        __m256 q_vec = _mm256_castsi256_ps(
                _mm256_and_si256(_mm256_castps_si256(_mm256_loadu_ps(&query_vec[94])), mask));

        __m256 diff_vec = d_vec - q_vec;
        diff_vec *= diff_vec;
        sum_vec += diff_vec;
    }

    std::cout << "simd final sum: " << std::setprecision(15) << mm256_hadd_ps(sum_vec) << std::endl;

    return mm256_hadd_ps(sum_vec);
}

int main()
{
    std::vector<float> vb{0.11232};

    for (int i = 1; i < 102; i++)
    {
        vb.push_back(vb[i - 1] * (i % 2 == 0 ? 1.321431 : -0.87382));
    }

    std::vector<float> vb2(vb.rbegin(), vb.rend());

    auto fvb = compare_with_id<float>(vb, vb2);
    auto fvb2 = dist_to_query(vb, vb2);

    std::cout << "\nAbsolute Difference: " << std::setprecision(15) << std::abs(fvb - fvb2) << "\nFinal Values: " << fvb
              << " " << fvb2 << std::endl;
}