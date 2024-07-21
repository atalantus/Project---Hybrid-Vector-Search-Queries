#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <array>
#include <immintrin.h>
#include <cstdint>
#include <thread>
#include <cmath>

/*
 * Enables performance debugging.
 */
#define ENABLE_PERF_DBG 1

/*
 * Defines the number of threads to be used.
 *
 * If 0 the number of threads will be set based on the workload size
 * and the CPU's thread count.
 */
#define NUM_THREADS 0

/*
 * Enables SIMD optimization for the find_worst operation.
 *
 * Does not have any effect on Heap Knn.
 */
#define FIND_WORST_SIMD 1

/*
 * Enables faster get_knn_sorted implementation.
 *
 * Does not have any effect on Heap Knn.
 */
#define SINGLE_SORTED 1

/*
 * Due to the order of operations changing when doing SIMD floating point math
 * the calculated distances between vectors won't be exactly the same as if
 * calculated without SIMD (smaller differences in the 5th decimal point).
 *
 * However, these changes can be big enough to yield a different result dataset
 * than the baseline implementation. See fp_inaccuracy_test.cpp
 */
#define DIST_SIMD 1
/*
 * With SIMD the distance calculation becomes so fast that any form of early bailout
 * check does not seem to yield any performance benefit given the performance hit
 * of the horizontal "hsum256_ps_avx" addition.
 */
#define DIST_BAIL_OUT 0

#include "optimized_impl.h"
#include "util.h"
#include "threading.hpp"

void vec_query(vector<vector<float>>& nodes, vector<vector<float>>& queries, float sample_proportion,
               vector<vector<uint32_t>>& knn_results)
{
    const uint32_t n = nodes.size();
    const uint32_t d = nodes[0].size();
    const uint32_t nq = queries.size();
    const uint32_t sn = uint32_t(sample_proportion * n);

    cout << "# data points:  " << n << "\n";
    cout << "# data point dim:  " << d << "\n";
    cout << "# queries:      " << nq << "\n";

#if NUM_THREADS
    const uint32_t thread_n = NUM_THREADS;
#else
    const uint32_t MIN_THREAD_WORKLOAD = 100000;
    const uint32_t thread_n = std::max(1U, std::min(std::thread::hardware_concurrency(), sn / MIN_THREAD_WORKLOAD));
#endif

    std::cerr << "Using " << thread_n << " threads\n";

    vector<Knn> knns;
    knns.reserve(thread_n);
    for (size_t i = 0; i < thread_n; ++i)
    {
        knns.emplace_back(nodes);
    }

    ThreadPool<Knn> thread_pool(thread_n, knns);

    for (uint i = 0; i < nq; i++)
    {
        const uint32_t query_type = queries[i][0];
        const int32_t v = queries[i][1];
        const float l = queries[i][2];
        const float r = queries[i][3];
        // skip first 2 elements to align with data vectorss
        q_vec_t query_vec(queries[i].begin() + 2, queries[i].end());

        thread_pool.parallel_for(sn, [&](uint32_t start, uint32_t end, Knn& knn)
        {
            knn.init(&query_vec);

            // Handling 4 types of queries
            if (query_type == 0)
            {  // only ANN
                for (uint32_t j = start; j < end; j++)
                {
                    knn.check_add(j);
                }
            } else if (query_type == 1)
            { // equal + ANN
                for (uint32_t j = start; j < end; j++)
                {
                    if (nodes[j][0] == v)
                    {
                        knn.check_add(j);
                    }
                }
            } else if (query_type == 2)
            { // range + ANN
                for (uint32_t j = start; j < end; j++)
                {
                    if (nodes[j][1] >= l && nodes[j][1] <= r)
                    {
                        knn.check_add(j);
                    }
                }
            } else if (query_type == 3)
            { // equal + range + ANN
                for (uint32_t j = start; j < end; j++)
                {
                    if (nodes[j][0] == v && nodes[j][1] >= l && nodes[j][1] <= r)
                    {
                        knn.check_add(j);
                    }
                }
            }
        });

        // merge knns
        auto final_knn = knns[0];
        for (uint32_t j = 1; j < thread_n; ++j)
        {
            final_knn.merge(knns[j]);
        }

        // If the number of knn in the sampled data is less than K, then fill the rest with the last few nodes
        if (final_knn.size() < KNN_LIMIT)
        {
            uint32_t s = 1;
            while (final_knn.size() < KNN_LIMIT)
            {
                final_knn.check_add(n - s);
                s = s + 1;
            }
        }

        knn_results.push_back(final_knn.get_knn_sorted());
    }

    PERF_DBG(
#if RDTSC
            auto unit = " cycles";
#else
            auto unit = " ns";
#endif

            std::cerr << "total dist calcs: \t" << dist_calcs << std::endl;
            std::cerr << "total bailouts: \t" << bailout << std::endl;
            std::cerr << "total dist calc: \t" << dist_calc_t << unit << "\t(per thread: " << dist_calc_t / thread_n << unit << ")" << std::endl;
            std::cerr << "total knn check_add: \t" << knn_check_t << unit << "\t(per thread: " << knn_check_t / thread_n << unit << ")" << std::endl;
            std::cerr << "total knn sort: \t" << knn_sort_t << unit << " \t(per thread: " << knn_sort_t / thread_n << unit << ")" << std::endl;
            std::cerr << "knn merge: \t\t" << knn_merge_t << unit << std::endl;
    )
}