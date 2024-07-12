#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <array>
#include <immintrin.h>
#include <cstdint>

#define ENABLE_PERF_DBG 1

#define USE_HEAP_KNN 0

#define FIND_WORST_SIMD 1

#define SINGLE_SORTED 1

/*
 * Due to the order of operations changing when doing SIMD floating point math
 * the calculated distances between vectors won't be exactly the same as if
 * calculated without SIMD (smaller differences in the 5th decimal point).
 *
 * However, these changes can be big enough to yield a different result dataset
 * than the baseline implementation.
 */
#define DIST_SIMD 1

/*
 * With SIMD the distance calculation becomes so fast (for "only" 100 dimensions at least)
 * that any form of early bailout check does not seem to yield any performance benefit.
 */
#define DIST_BAIL_OUT 0

#include "optimized_impl.h"
#include "util.h"

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

#if USE_HEAP_KNN
    KnnHeap knn(nodes);
#else
    Knn knn(nodes);
#endif

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
                {
                    knn.check(j);
                }
            }
        } else if (query_type == 3)
        { // equal + range + ANN
            for (uint32_t j = 0; j < sn; j++)
            {
                if (nodes[j][0] == v && nodes[j][1] >= l && nodes[j][1] <= r)
                {
                    knn.check(j);
                }
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

    PERF_DBG(
            std::cerr << "dist calc:\t" << dist_calc_t << std::endl;
            std::cerr << "knn check:\t" << knn_check_t << std::endl;
            std::cerr << "knn sort:\t" << knn_sort_t << std::endl;
    )
}