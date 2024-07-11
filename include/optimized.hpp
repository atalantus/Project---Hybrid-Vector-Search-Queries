#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <array>
#include <immintrin.h>
#include <cstdint>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::array;

typedef vector<float> d_vec_t;
typedef vector<float> q_vec_t;

constexpr size_t KNN_LIMIT = 100;

#define FIND_WORST_SIMD 1

/*
 * Due to the order of operations changing when doing SIMD floating point math
 * the calculated distances between vectors won't be exactly the same as if
 * calculated without SIMD (smaller differences in the 5th decimal point).
 *
 * However, these changes can be big enough to yield a different result dataset
 * than the baseline implementation.
 */
#define DIST_SIMD 0

#define DIST_BAIL_OUT 0

// very efficient horizontal add for eight 32-bit floats in a 256-bit register
// courtesy of: https://stackoverflow.com/a/13222410/6920681
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

float dist_to_query(const d_vec_t& data_vec, const q_vec_t& query_vec, [[maybe_unused]] float worst)
{
#if DIST_SIMD

    __m256 sum_vec = _mm256_set1_ps(0.0);

#if DIST_BAIL_OUT

    // Skip the first 2 dimensions
    size_t i = 2;
    for (; i < 42; i += 8)
    {
        __m256 d_vec = _mm256_loadu_ps(&data_vec[i]);
        __m256 q_vec = _mm256_loadu_ps(&query_vec[i]);

        __m256 diff_vec = d_vec - q_vec;
        diff_vec *= diff_vec;
        sum_vec += diff_vec;
    }

    // check for early bailout
    auto cur_sum = mm256_hadd_ps(sum_vec);
    if (cur_sum >= worst)
    {
        return std::numeric_limits<float>::infinity();
    }

    for (; i < 98; i += 8)
    {
        __m256 d_vec = _mm256_loadu_ps(&data_vec[i]);
        __m256 q_vec = _mm256_loadu_ps(&query_vec[i]);

        __m256 diff_vec = d_vec - q_vec;
        diff_vec *= diff_vec;
        sum_vec += diff_vec;
    }

#else

    // Skip the first 2 dimensions
    for (size_t i = 2; i < 98; i += 8)
    {
        __m256 d_vec = _mm256_loadu_ps(&data_vec[i]);
        __m256 q_vec = _mm256_loadu_ps(&query_vec[i]);

        __m256 diff_vec = d_vec - q_vec;
        diff_vec *= diff_vec;
        sum_vec += diff_vec;
    }

#endif

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

    return mm256_hadd_ps(sum_vec);

#else
#if DIST_BAIL_OUT

    float sum = 0.0;

    // Skip the first 2 dimensions
    size_t i = 2;
    for (; i < 52; ++i)
    {
        float diff = data_vec[i] - query_vec[i];
        sum += diff * diff;
    }

    // check for early bailout
    if (sum >= worst)
    {
        return std::numeric_limits<float>::infinity();
    }

    for (; i < 102; ++i)
    {
        float diff = data_vec[i] - query_vec[i];
        sum += diff * diff;
    }

    return sum;

#else

    float sum = 0.0;
    // Skip the first 2 dimensions
    for (size_t i = 2; i < 102; ++i)
    {
        float diff = data_vec[i] - query_vec[i];
        sum += diff * diff;
    }

    return sum;

#endif
#endif
}

class Knn
{
private:
    bool is_full;
    uint32_t fill;
    uint32_t worst;

    q_vec_t* query_vec;

    alignas(32) array<float, KNN_LIMIT> dist_array{};
    alignas(32) array<uint32_t, KNN_LIMIT> vec_idx_array{};

    const vector<d_vec_t>& _nodes;

public:
    explicit Knn(const vector<d_vec_t>& nodes) : _nodes(nodes)
    {
        fill = 0;
        is_full = false;
        worst = 0;
        query_vec = nullptr;
    }

private:
    inline void find_worst()
    {
        assert(fill == KNN_LIMIT);

#if FIND_WORST_SIMD

        __m256 cur_worst_dist_vec = _mm256_loadu_ps(&dist_array[0]);
        __m256i cur_idx_vec = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        __m256i cur_worst_idx_vec = cur_idx_vec;

        for (int i = 8; i < 96; i += 8)
        {
            __m256 cur_dist_vec = _mm256_loadu_ps(&dist_array[i]);
            cur_idx_vec = _mm256_set_epi32(i + 7, i + 6, i + 5, i + 4, i + 3, i + 2, i + 1, i);

            __m256 cmp_lt = _mm256_cmp_ps(cur_dist_vec, cur_worst_dist_vec, _CMP_GT_OQ);

            cur_worst_dist_vec = _mm256_blendv_ps(cur_worst_dist_vec, cur_dist_vec, cmp_lt);
            cur_worst_idx_vec = _mm256_blendv_epi8(cur_worst_idx_vec, cur_idx_vec, _mm256_castps_si256(cmp_lt));
        }

        // also do the remaining elements
        {
            __m256 cur_dist_vec = _mm256_loadu_ps(&dist_array[92]);
            cur_idx_vec = _mm256_set_epi32(99, 98, 97, 96, 95, 94, 93, 92);

            __m256 cmp_lt = _mm256_cmp_ps(cur_dist_vec, cur_worst_dist_vec, _CMP_GT_OQ);

            cur_worst_dist_vec = _mm256_blendv_ps(cur_worst_dist_vec, cur_dist_vec, cmp_lt);
            cur_worst_idx_vec = _mm256_blendv_epi8(cur_worst_idx_vec, cur_idx_vec, _mm256_castps_si256(cmp_lt));
        }

        float worst_distances[8];
        uint32_t worst_indices[8];

        _mm256_storeu_ps(&worst_distances[0], cur_worst_dist_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(&worst_indices[0]), cur_worst_idx_vec);

        float worst_dist = worst_distances[0];
        worst = worst_indices[0];
        for (int i = 1; i < 8; ++i)
        {
            if (worst_distances[i] > worst_dist)
            {
                worst_dist = worst_distances[i];
                worst = worst_indices[i];
            }
        }

#else

        float cur_worst_dist = dist_array[0];
        worst = 0;

        for (int i = 0; i < KNN_LIMIT; ++i)
        {
            if (dist_array[i] > cur_worst_dist)
            {
                cur_worst_dist = dist_array[i];
                worst = i;
            }
        }

#endif
    }

public:
    void init(q_vec_t* query_vector)
    {
        fill = 0;
        is_full = false;
        worst = 0;
        query_vec = query_vector;
    }

    inline void check(uint32_t vec_idx)
    {
        float worst_dist = dist_array[worst];
        float bailout_dist = is_full ? worst_dist : std::numeric_limits<float>::infinity();
        float dist = dist_to_query(_nodes[vec_idx], *query_vec, bailout_dist);

        if (!is_full)
        {
            // insert at the back
            worst = dist < worst_dist ? worst : fill;
            dist_array[fill] = dist;
            vec_idx_array[fill] = vec_idx;
            ++fill;
            is_full = fill == KNN_LIMIT;
        } else if (dist < worst_dist)
        {
            // replace worst with new element and find new worst
            dist_array[worst] = dist;
            vec_idx_array[worst] = vec_idx;
            find_worst();
            return;
        } else
        {
            // new element is not better than the worst element in array
            return;
        }
    }

    [[nodiscard]] inline uint32_t size() const
    {
        return fill;
    }

    inline vector<uint32_t> get_knn_sorted()
    {
        vector<uint32_t> ids;
        ids.resize(fill);
        std::iota(ids.begin(), ids.end(), 0);
        std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b)
        {
            return dist_array[a] < dist_array[b];
        });

        vector<uint32_t> knn_sorted;
        knn_sorted.resize(fill);

        for (int i = 0; i < fill; ++i)
        {
            knn_sorted[i] = vec_idx_array[ids[i]];
        }

        return knn_sorted;
    }
};

void vec_query(vector<vector<float>>& nodes, vector<vector<float>>& queries, float sample_proportion,
               vector<vector<uint32_t>>& knn_results)
{
    uint32_t n = nodes.size();
    uint32_t d = nodes[0].size();
    uint32_t nq = queries.size();
    uint32_t sn = uint32_t(sample_proportion * n);

    cout << "# data points:  " << n << "\n";
    cout << "# data point dim:  " << d << "\n";
    cout << "# queries:      " << nq << "\n";

    /** A basic method to compute the KNN results using sampling  **/
    Knn knn(nodes);

    for (uint i = 0; i < nq; i++)
    {
        uint32_t query_type = queries[i][0];
        int32_t v = queries[i][1];
        float l = queries[i][2];
        float r = queries[i][3];
        // skip first 2 elements to align with data vectors
        q_vec_t query_vec(queries[i].begin() + 2, queries[i].end());

        knn.init(&query_vec);

        // Handling 4 types of queries
        if (query_type == 0)
        {  // only ANN
            for (uint32_t j = 0; j < sn; j++)
            {
                knn.check(j);
            }
        } else if (query_type == 1)
        { // equal + ANN
            for (uint32_t j = 0; j < sn; j++)
            {
                if (nodes[j][0] == v)
                {
                    knn.check(j);
                }
            }
        } else if (query_type == 2)
        { // range + ANN
            for (uint32_t j = 0; j < sn; j++)
            {
                if (nodes[j][1] >= l && nodes[j][1] <= r)
                    knn.check(j);
            }
        } else if (query_type == 3)
        { // equal + range + ANN
            for (uint32_t j = 0; j < sn; j++)
            {
                if (nodes[j][0] == v && nodes[j][1] >= l && nodes[j][1] <= r)
                    knn.check(j);
            }
        }

        // If the number of knn in the sampled data is less than K, then fill the rest with the last few nodes
        if (knn.size() < KNN_LIMIT)
        {
            uint32_t s = 1;
            while (knn.size() < KNN_LIMIT)
            {
                knn.check(n - s);
                s = s + 1;
            }
        }

        knn_results.push_back(knn.get_knn_sorted());
    }
}