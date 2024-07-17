#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>

/*
 * Enables performance debugging.
 */
#define ENABLE_PERF_DBG 0

#include "optimized_impl.h"
#include "util.h"

#if 0

float compare_with_id(const d_vec_t& data_vec, const q_vec_t& query_vec)
{
    __m256 sum_vec = _mm256_set1_ps(0.0);

    // Skip the first 2 dimensions
    for (size_t i = 2; i < VEC_DIM - (VEC_DIM % 8) + 2; i += 8)
    {
        __m256 d_vec = _mm256_loadu_ps(&data_vec[i]);
        __m256 q_vec = _mm256_loadu_ps(&query_vec[i]);

        __m256 diff_vec = d_vec - q_vec;
        diff_vec *= diff_vec;
        sum_vec += diff_vec;
    }

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
}

#else

float compare_with_id(const std::vector<float>& a, const std::vector<float>& b)
{
    float sum = 0.0;
    // Skip the first 2 dimensions
    for (size_t i = 2; i < VEC_DIM; ++i)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}
#endif

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

    PERF_DBG(
            uint64_t index_t = 0;
            uint64_t dist_t = 0;
            uint64_t sort_t = 0;
    )

    for (uint i = 0; i < nq; i++)
    {
        uint32_t query_type = queries[i][0];
        int32_t v = queries[i][1];
        float l = queries[i][2];
        float r = queries[i][3];
        vector<float> query_vec;

        // first push_back 2 zeros for aligning with dataset
        query_vec.push_back(0);
        query_vec.push_back(0);
        for (uint j = 4; j < queries[i].size(); j++)
            query_vec.push_back(queries[i][j]);

        vector<uint32_t> knn; // candidate knn

        PERF_DBG(auto s1 = rdtsc();)

        // Handling 4 types of queries
        if (query_type == 0)
        {  // only ANN
            for (uint32_t j = 0; j < sn; j++)
            {
                knn.push_back(j);
            }
        } else if (query_type == 1)
        { // equal + ANN
            for (uint32_t j = 0; j < sn; j++)
            {
                if (nodes[j][0] == v)
                {
                    knn.push_back(j);
                }
            }
        } else if (query_type == 2)
        { // range + ANN
            for (uint32_t j = 0; j < sn; j++)
            {
                if (nodes[j][1] >= l && nodes[j][1] <= r)
                    knn.push_back(j);
            }
        } else if (query_type == 3)
        { // equal + range + ANN
            for (uint32_t j = 0; j < sn; j++)
            {
                if (nodes[j][0] == v && nodes[j][1] >= l && nodes[j][1] <= r)
                    knn.push_back(j);
            }
        }

        // If the number of knn in the sampled data is less than K, then fill the rest with the last few nodes
        if (knn.size() < KNN_LIMIT)
        {
            uint32_t s = 1;
            while (knn.size() < KNN_LIMIT)
            {
                knn.push_back(n - s);
                s = s + 1;
            }
        }

        PERF_DBG(auto s2 = rdtsc();index_t += s2 - s1;)

        // build another vec to store the distance between knn[i] and query_vec
        vector<float> dists;
        dists.resize(knn.size());
        for (uint32_t j = 0; j < knn.size(); j++)
            dists[j] = compare_with_id(nodes[knn[j]], query_vec);

        PERF_DBG(auto s3 = rdtsc();dist_t += s3 - s2;)

        vector<uint32_t> ids;
        ids.resize(knn.size());
        std::iota(ids.begin(), ids.end(), 0);
        // sort ids based on dists
        std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b)
        {
            return dists[a] < dists[b];
        });
        vector<uint32_t> knn_sorted;
        knn_sorted.resize(KNN_LIMIT);
        for (uint32_t j = 0; j < KNN_LIMIT; j++)
        {
            knn_sorted[j] = knn[ids[j]];
        }

        PERF_DBG(auto s4 = rdtsc();sort_t += s4 - s3;)

        knn_results.push_back(knn_sorted);
    }

    PERF_DBG(
            std::cerr << "indexing cycles:\t" << index_t << std::endl;
            std::cerr << "dist calc cycles:\t" << dist_t << std::endl;
            std::cerr << "sorting cycles:\t" << sort_t << std::endl;
    )
}