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
#include "util.h"
#include <limits>
#include <atomic>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::array;

typedef vector<float> d_vec_t;
typedef vector<float> q_vec_t;

constexpr size_t KNN_LIMIT = 100;

constexpr size_t VEC_DIM = 102;

static_assert(KNN_LIMIT >= 8);
static_assert(VEC_DIM >= 10);

/*
 * Fastest way to horizontally add floats of a 256-bit register.
 * Courtesy of: https://stackoverflow.com/a/35270026/6920681
 */
float hsum256_ps_avx(__m256 v)
{
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    vlow = vlow + vhigh;     // add the low 128
    __m128 shuf = _mm_movehdup_ps(vlow);        // broadcast elements 3,1 to 2,0
    __m128 sums = vlow + shuf;
    shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums = sums + shuf;
    return _mm_cvtss_f32(sums);
}

PERF_DBG(
        std::atomic<uint64_t> dist_calcs = 0;
        std::atomic<uint64_t> bailout = 0;
)

float dist_to_query(const d_vec_t& data_vec, const q_vec_t& query_vec, [[maybe_unused]] float worst)
{
    PERF_DBG(++dist_calcs;)

#if DIST_SIMD

    __m256 sum_vec = _mm256_set1_ps(0.0);

#if DIST_BAIL_OUT

    // Skip the first 2 dimensions
    size_t i = 2;
    for (; i < (VEC_DIM / 4 * 3) + 2; i += 8)
    {
        __m256 d_vec = _mm256_loadu_ps(&data_vec[i]);
        __m256 q_vec = _mm256_loadu_ps(&query_vec[i]);

        __m256 diff_vec = d_vec - q_vec;
        diff_vec *= diff_vec;
        sum_vec += diff_vec;
    }

    // check for bailout
    auto cur_sum = hsum256_ps_avx(sum_vec);
    if (cur_sum >= worst)
    {
        PERF_DBG(bailout++;)
        return std::numeric_limits<float>::infinity();
    }

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
        auto r = (VEC_DIM - 2) % 8;
        auto cm = [r](size_t n) -> int
        { return n <= r ? -1 : 0; };
        __m256i mask = _mm256_set_epi32(cm(1), cm(2), cm(3), cm(4), cm(5), cm(6), cm(7), 0);
        __m256 d_vec = _mm256_castsi256_ps(
                _mm256_and_si256(_mm256_castps_si256(_mm256_loadu_ps(&data_vec[VEC_DIM - 8])), mask));
        __m256 q_vec = _mm256_castsi256_ps(
                _mm256_and_si256(_mm256_castps_si256(_mm256_loadu_ps(&query_vec[VEC_DIM - 8])), mask));

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
    // bailout at around 75% percent of the sum
    for (; i < VEC_DIM / 4 * 3; ++i)
    {
        float diff = data_vec[i] - query_vec[i];
        sum += diff * diff;
    }

    // check for early bailout
    if (sum >= worst)
    {
        PERF_DBG(bailout++;)
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

PERF_DBG(
        std::atomic<uint64_t> dist_calc_t = 0;
        std::atomic<uint64_t> knn_check_t = 0;
        std::atomic<uint64_t> find_worst_t = 0;
        std::atomic<uint64_t> knn_sort_t = 0;
        std::atomic<uint64_t> knn_merge_t = 0;
)

class alignas(64) Knn
{
private:
    q_vec_t* query_vec;

    const vector<d_vec_t>& _nodes;

    array<float, KNN_LIMIT> dist_array{};
    array<uint32_t, KNN_LIMIT> node_idx_array{};

    uint32_t fill;
    uint32_t worst;

public:
    explicit Knn(const vector<d_vec_t>& nodes) : _nodes(nodes)
    {
        fill = 0;
        worst = 0;
        query_vec = nullptr;
    }

private:
    inline uint32_t find_worst()
    {
        assert(fill == KNN_LIMIT);

#if FIND_WORST_SIMD

        __m256 cur_worst_dist_vec = _mm256_loadu_ps(&dist_array[0]);
        __m256i cur_idx_vec = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        __m256i idx_add_vec = _mm256_set1_epi32(8);
        __m256i cur_worst_idx_vec = cur_idx_vec;

        for (size_t i = 8; i < KNN_LIMIT - (KNN_LIMIT % 8); i += 8)
        {
            __m256 cur_dist_vec = _mm256_loadu_ps(&dist_array[i]);
            cur_idx_vec += idx_add_vec;

            __m256 cmp_lt = _mm256_cmp_ps(cur_dist_vec, cur_worst_dist_vec, _CMP_GT_OQ);

            cur_worst_dist_vec = _mm256_blendv_ps(cur_worst_dist_vec, cur_dist_vec, cmp_lt);
            cur_worst_idx_vec = _mm256_blendv_epi8(cur_worst_idx_vec, cur_idx_vec, _mm256_castps_si256(cmp_lt));
        }

        // also do the remaining elements
        {
            cur_idx_vec = _mm256_set_epi32(KNN_LIMIT - 1, KNN_LIMIT - 2, KNN_LIMIT - 3, KNN_LIMIT - 4, KNN_LIMIT - 5,
                                           KNN_LIMIT - 6, KNN_LIMIT - 7, KNN_LIMIT - 8);
            __m256 cur_dist_vec = _mm256_loadu_ps(&dist_array[KNN_LIMIT - 8]);

            __m256 cmp_lt = _mm256_cmp_ps(cur_dist_vec, cur_worst_dist_vec, _CMP_GT_OQ);

            cur_worst_dist_vec = _mm256_blendv_ps(cur_worst_dist_vec, cur_dist_vec, cmp_lt);
            cur_worst_idx_vec = _mm256_blendv_epi8(cur_worst_idx_vec, cur_idx_vec, _mm256_castps_si256(cmp_lt));
        }

        // Step 1: Find the maximum distance using SIMD
        __m256 max_dist = cur_worst_dist_vec;
        max_dist = _mm256_max_ps(max_dist, _mm256_permute2f128_ps(max_dist, max_dist, 1));
        max_dist = _mm256_max_ps(max_dist, _mm256_permute_ps(max_dist, 0x4E));
        max_dist = _mm256_max_ps(max_dist, _mm256_permute_ps(max_dist, 0xB1));

        // Step 2: Create a mask for the maximum values
        __m256 mask = _mm256_cmp_ps(cur_worst_dist_vec, max_dist, _CMP_EQ_OQ);

        // Step 3: Use the mask to select the corresponding indices
        __m256i selected_indices = _mm256_blendv_epi8(_mm256_setzero_si256(), cur_worst_idx_vec,
                                                      _mm256_castps_si256(mask));

        // Step 4: Shuffle the selected index into lower 32 bit
        selected_indices = _mm256_max_epu32(selected_indices,
                                            _mm256_permute2f128_si256(selected_indices, selected_indices, 1));
        selected_indices = _mm256_max_epu32(selected_indices, _mm256_shuffle_epi32(selected_indices, 0x4E));
        selected_indices = _mm256_max_epu32(selected_indices, _mm256_shuffle_epi32(selected_indices, 0xB1));

        // Extract results
        return _mm_cvtsi128_si32(_mm256_castsi256_si128(selected_indices));

#else

        float cur_worst_dist = dist_array[0];
        uint32_t worst = 0;

        for (int i = 0; i < KNN_LIMIT; ++i)
        {
            if (dist_array[i] > cur_worst_dist)
            {
                cur_worst_dist = dist_array[i];
                worst = i;
            }
        }

        return worst;

#endif
    }

public:
    void init(q_vec_t* query_vector)
    {
        fill = 0;
        worst = 0;
        query_vec = query_vector;
    }

    inline void check_add(uint32_t node_idx)
    {
        const bool not_full = fill < KNN_LIMIT;
        const float worst_dist = dist_array[worst];
        const float bailout_dist = not_full ? std::numeric_limits<float>::infinity() : worst_dist;
        PERF_DBG(auto s1 = rdtsc();)
        const float dist = dist_to_query(_nodes[node_idx], *query_vec, bailout_dist);
        PERF_DBG(auto s2 = rdtsc();dist_calc_t += s2 - s1;)

        /*
         * The amount of cycles spent in this section is astonishing.
         * I thought this to be the result of branch miss predictions, however,
         * the (complex) branchless code below did neither result in any performance
         * benefits nor (noticeably) less branch misses overall.
         */
#if 1

        const bool better_than_worst = dist < worst_dist;
        const bool add_new_vec = not_full || better_than_worst;
        const uint32_t update_idx = not_full ? fill : worst;
        fill += not_full;

        dist_array[update_idx] = add_new_vec ? dist : worst_dist;
        node_idx_array[update_idx] = add_new_vec ? node_idx : node_idx_array[worst];

        worst = better_than_worst ? worst : update_idx;
        worst = (better_than_worst && !not_full) ? find_worst() : worst;

#else

        if (__builtin_expect(not_full, 0))
        {
            // insert at the back
            worst = dist < worst_dist ? worst : fill;
            dist_array[fill] = dist;
            node_idx_array[fill] = node_idx;
            ++fill;
        } else if (dist < worst_dist)
        {
            // replace worst with new element and find new worst
            dist_array[worst] = dist;
            node_idx_array[worst] = node_idx;
            worst = find_worst();
        } else
        {
            // new element is not better than the worst element in array
        }

#endif

        PERF_DBG(knn_check_t += rdtsc() - s2;)
    }

    inline void merge(Knn& other)
    {
        for (uint32_t i = 0; i < other.fill; ++i)
        {
            const bool not_full = fill < KNN_LIMIT;
            const float worst_dist = dist_array[worst];
            const float other_dist = other.dist_array[i];
            PERF_DBG(auto s2 = rdtsc();)

#if 1

            const bool better_than_worst = other_dist < worst_dist;
            const bool add_new_vec = not_full || better_than_worst;
            const uint32_t update_idx = not_full ? fill : worst;
            fill += not_full;

            dist_array[update_idx] = add_new_vec ? other_dist : worst_dist;
            node_idx_array[update_idx] = add_new_vec ? other.node_idx_array[i] : node_idx_array[worst];

            worst = better_than_worst ? worst : update_idx;
            worst = (better_than_worst && !not_full) ? find_worst() : worst;

#else

            if (__builtin_expect(not_full, 0))
            {
                // insert at the back
                worst = other_dist < worst_dist ? worst : fill;
                dist_array[fill] = other_dist;
                node_idx_array[fill] = other.node_idx_array[i];
                ++fill;
            } else if (other_dist < worst_dist)
            {
                // replace worst with new element and find new worst
                dist_array[worst] = other_dist;
                node_idx_array[worst] = other.node_idx_array[i];
                worst = find_worst();
            } else
            {
                // new element is not better than the worst element in array
            }

#endif

            PERF_DBG(knn_merge_t += rdtsc() - s2;)
        }
    }

    [[nodiscard]] inline uint32_t size() const
    {
        return fill;
    }

    inline vector<uint32_t> get_knn_sorted()
    {
        PERF_DBG(auto s = rdtsc();)

        vector<uint32_t> knn_sorted;
        knn_sorted.resize(fill);

#if SINGLE_SORTED

        std::array<std::pair<float, uint32_t>, KNN_LIMIT> sorted_knn;

        for (uint32_t i = 0; i < fill; ++i)
        {
            sorted_knn[i] = {dist_array[i], node_idx_array[i]};
        }

        std::sort(sorted_knn.begin(), sorted_knn.begin() + fill,
                  [](const auto& a, const auto& b)
                  { return a.first < b.first; });

        for (uint32_t i = 0; i < fill; ++i)
        {
            knn_sorted[i] = sorted_knn[i].second;
        }

#else

        vector<uint32_t> ids;
        ids.resize(fill);
        std::iota(ids.begin(), ids.end(), 0);
        std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b)
        {
            return dist_array[a] < dist_array[b];
        });

        for (int i = 0; i < fill; ++i)
        {
            knn_sorted[i] = node_idx_array[ids[i]];
        }

#endif

        PERF_DBG(knn_sort_t += rdtsc() - s;)

        return knn_sorted;
    }
};

class alignas(64) KnnHeap
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

    inline void check_add(uint32_t vec_idx)
    {
        float worst_dist = knn_heap.front().dist;
        float bailout_dist = is_full ? worst_dist : std::numeric_limits<float>::infinity();
        PERF_DBG(auto s = rdtsc();)
        float dist = dist_to_query(_nodes[vec_idx], *query_vec, bailout_dist);
        PERF_DBG(auto s2 = rdtsc();dist_calc_t += s2 - s;)

        if (__builtin_expect(!is_full, 0))
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
        } else
        {
            // new element is not better than the worst element in array
        }

        PERF_DBG(knn_check_t += rdtsc() - s2;)
    }

    [[nodiscard]] inline uint32_t size() const
    {
        return fill;
    }

    inline vector<uint32_t> get_knn_sorted()
    {
        PERF_DBG(auto s = rdtsc();)
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

        PERF_DBG(knn_sort_t += rdtsc() - s;)

        return knn_sorted;
    }
};