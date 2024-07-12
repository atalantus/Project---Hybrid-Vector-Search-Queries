#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <array>
#include <immintrin.h>
#include <functional>
#include <cstdint>
#include <cassert>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::array;

typedef vector<float> d_vec_t;
typedef vector<float> q_vec_t;

constexpr size_t KNN_LIMIT = 100; // at least 8

constexpr size_t VEC_DIM = 102; // at least 8

/*
 * Fastest way to horizontally add floats of a 256-bit register.
 * Courtesy of: https://stackoverflow.com/a/35270026/6920681
 */
float hsum256_ps_avx(__m256 v)
{
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

float dist_to_query(const d_vec_t& data_vec, const q_vec_t& query_vec, [[maybe_unused]] float worst)
{
#if DIST_SIMD

    __m256 sum_vec = _mm256_set1_ps(0.0);

#if DIST_BAIL_OUT

    // Skip the first 2 dimensions
    size_t i = 2;
    for (; i < (VEC_DIM / 16 * 8) + 2; i += 8)
    {
        __m256 d_vec = _mm256_loadu_ps(&data_vec[i]);
        __m256 q_vec = _mm256_loadu_ps(&query_vec[i]);

        __m256 diff_vec = d_vec - q_vec;
        diff_vec *= diff_vec;
        sum_vec += diff_vec;
    }

    // bailout via horizontal sum
    auto cur_sum = hsum256_ps_avx(sum_vec);
    if (cur_sum >= worst)
    {
        return std::numeric_limits<float>::infinity();
    }

    // bailout via sum vector comparison
//    __m256 bailout_vec = _mm256_set1_ps(worst);
//    __m256i bailout_cmp_vec = _mm256_castps_si256(_mm256_cmp_ps(sum_vec, bailout_vec, _CMP_GE_OQ));
//    if (_mm256_testz_si256(bailout_cmp_vec, bailout_cmp_vec) == 0)
//    {
//        bailouts++;
//        return std::numeric_limits<float>::infinity();
//    }

    for (; i < VEC_DIM - (VEC_DIM % 8) + 2; i += 8)
    {
        __m256 d_vec = _mm256_loadu_ps(&data_vec[i]);
        __m256 q_vec = _mm256_loadu_ps(&query_vec[i]);

        __m256 diff_vec = d_vec - q_vec;
        diff_vec *= diff_vec;
        sum_vec += diff_vec;
    }

#else

    // Skip the first 2 dimensions
    for (size_t i = 2; i < VEC_DIM - (VEC_DIM % 8) + 2; i += 8)
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
        auto r = VEC_DIM % 8;
        auto cm = [r](size_t n) -> int { return n >= r ? -1 : 0;};
        __m256i mask = _mm256_set_epi32(cm(1), cm(2), cm(3), cm(4), cm(5), cm(6), cm(7), 0);
        __m256 d_vec = _mm256_castsi256_ps(
                _mm256_and_si256(_mm256_castps_si256(_mm256_loadu_ps(&data_vec[VEC_DIM - (VEC_DIM % 8)])), mask));
        __m256 q_vec = _mm256_castsi256_ps(
                _mm256_and_si256(_mm256_castps_si256(_mm256_loadu_ps(&query_vec[VEC_DIM - (VEC_DIM % 8)])), mask));

        __m256 diff_vec = d_vec - q_vec;
        diff_vec *= diff_vec;
        sum_vec += diff_vec;
    }

    return hsum256_ps_avx(sum_vec);

#else
#if DIST_BAIL_OUT

    float sum = 0.0;

    // Skip the first 2 dimensions
    size_t i = 2;
    for (; i < VEC_DIM / 2; ++i)
    {
        float diff = data_vec[i] - query_vec[i];
        sum += diff * diff;
    }

    // check for early bailout
    if (sum >= worst)
    {
        return std::numeric_limits<float>::infinity();
    }

    for (; i < VEC_DIM; ++i)
    {
        float diff = data_vec[i] - query_vec[i];
        sum += diff * diff;
    }

    return sum;

#else

    float sum = 0.0;
    // Skip the first 2 dimensions
    for (size_t i = 2; i < VEC_DIM; ++i)
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

        for (int i = 8; i < KNN_LIMIT - (KNN_LIMIT % 8); i += 8)
        {
            __m256 cur_dist_vec = _mm256_loadu_ps(&dist_array[i]);
            cur_idx_vec = _mm256_set_epi32(i + 7, i + 6, i + 5, i + 4, i + 3, i + 2, i + 1, i);

            __m256 cmp_lt = _mm256_cmp_ps(cur_dist_vec, cur_worst_dist_vec, _CMP_GT_OQ);

            cur_worst_dist_vec = _mm256_blendv_ps(cur_worst_dist_vec, cur_dist_vec, cmp_lt);
            cur_worst_idx_vec = _mm256_blendv_epi8(cur_worst_idx_vec, cur_idx_vec, _mm256_castps_si256(cmp_lt));
        }

        // also do the remaining elements
        {
            __m256 cur_dist_vec = _mm256_loadu_ps(&dist_array[KNN_LIMIT - 8]);
            cur_idx_vec = _mm256_set_epi32(KNN_LIMIT - 1, KNN_LIMIT - 2, KNN_LIMIT - 3, KNN_LIMIT - 4, KNN_LIMIT - 5,
                                           KNN_LIMIT - 6, KNN_LIMIT - 7, KNN_LIMIT - 8);

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

class KnnHeap
{
private:
    bool is_full;
    uint32_t fill;

    q_vec_t* query_vec;

    struct knn_heap_el
    {
        float dist;
        uint32_t node_idx;

        knn_heap_el() : dist{0}, node_idx{0}
        {}

        knn_heap_el(float d, uint32_t i) : dist{d}, node_idx{i}
        {}
    };

    std::function<bool(const knn_heap_el& a, const knn_heap_el& b)> compare_heap_el = [](const knn_heap_el& a,
                                                                                         const knn_heap_el& b)
    { return a.dist < b.dist; };

    array<knn_heap_el, KNN_LIMIT> knn_heap;

    const vector<d_vec_t>& _nodes;

public:
    explicit KnnHeap(const vector<d_vec_t>& nodes) : _nodes(nodes)
    {
        fill = 0;
        is_full = false;
        query_vec = nullptr;
    }

public:
    void init(q_vec_t* query_vector)
    {
        fill = 0;
        is_full = false;
        query_vec = query_vector;
    }

    inline void check(uint32_t vec_idx)
    {
        float worst_dist = knn_heap.front().dist;
        float bailout_dist = is_full ? worst_dist : std::numeric_limits<float>::infinity();
        float dist = dist_to_query(_nodes[vec_idx], *query_vec, bailout_dist);

        if (!is_full)
        {
            // insert at the back
            knn_heap[fill++] = {dist, vec_idx};
            std::push_heap(knn_heap.begin(), knn_heap.begin() + fill, compare_heap_el);
            is_full = fill == KNN_LIMIT;

            assert(std::is_heap(knn_heap.begin(), knn_heap.begin() + fill, compare_heap_el));
        } else if (dist < worst_dist)
        {
            assert(fill == KNN_LIMIT);
            // replace worst with new element and find new worst
            std::pop_heap(knn_heap.begin(), knn_heap.end(), compare_heap_el);
            knn_heap.back().dist = dist;
            knn_heap.back().node_idx = vec_idx;
            std::push_heap(knn_heap.begin(), knn_heap.end(), compare_heap_el);

            assert(std::is_heap(knn_heap.begin(), knn_heap.end(), compare_heap_el));
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
        assert(std::is_heap(knn_heap.begin(), knn_heap.begin() + fill, compare_heap_el));

        vector<uint32_t> knn_sorted;
        knn_sorted.resize(fill);

        for (uint32_t i = fill; i > 0; --i)
        {
            knn_sorted[i - 1] = knn_heap.front().node_idx;
            std::pop_heap(knn_heap.begin(), knn_heap.begin() + i, compare_heap_el);
        }

        is_full = false;
        fill = 0;

        return knn_sorted;
    }
};