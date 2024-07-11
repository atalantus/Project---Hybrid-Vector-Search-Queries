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

        std::cout << "simd after " << i + 8 << " steps, sum: " << std::setprecision(15) << hsum256_ps_avx(sum_vec)
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

    std::cout << "simd final sum: " << std::setprecision(15) << hsum256_ps_avx(sum_vec) << std::endl;

    return hsum256_ps_avx(sum_vec);
}

int main()
{
    std::vector<float> vb{0.11232};
    std::vector<double> vd{0.11232};
    for (int i = 1; i < 102; i++)
    {
        vb.push_back(vb[i - 1] * (i % 2 == 0 ? 1.321431 : -0.87382));
        vd.push_back(vd[i - 1] * (i % 2 == 0 ? 1.321431 : -0.87382));
    }

    std::vector<float> vb2(vb.rbegin(), vb.rend());
    std::vector<double> vd2(vd.rbegin(), vd.rend());

    // The mathematically correct sum would be:
    auto fvb = compare_with_id<float>(vb, vb2);
    auto fvb2 = dist_to_query(vb, vb2);

    auto fvd = compare_with_id<double>(vd, vd2);

    std::cout << "\nAbsolute Difference: " << std::setprecision(15) << std::abs(fvb - fvb2) << "\nFinal Values: " << fvb
              << " " << fvb2 << " " << fvd << std::endl;
}