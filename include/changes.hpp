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

#define FIND_WORST_SIMD 0

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

float dist_to_query(const d_vec_t& data_vec, const q_vec_t& query_vec)
{
    float sum = 0.0;
    // Skip the first 2 dimensions
    // TODO: SIMD
    for (size_t i = 2; i < data_vec.size(); ++i)
    {
        float diff = data_vec[i] - query_vec[i];
        sum += diff * diff;
    }
    return sum;
}

class Knn
{
private:
    const vector<d_vec_t>& _nodes;
    q_vec_t* query_vec;

    uint32_t fill;
    uint32_t worst;

    alignas(32) array<float, 100> dist_array{};
    alignas(32) array<uint32_t, 100> vec_idx_array{};

public:
    explicit Knn(const vector<d_vec_t>& nodes) : _nodes(nodes)
    {
        fill = 0;
        worst = 0;
        query_vec = nullptr;
    }

private:
    inline void find_worst()
    {
        assert(fill == 100);

#if FIND_WORST_SIMD

        __m256 cur_worst_dist_vec = _mm256_load_ps(&dist_array[0]);
        __m256i cur_worst_idx_vec = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

        for (int i = 8; i < 96; i += 8)
        {
            __m256 cur_dist_vec = _mm256_load_ps(&dist_array[i]);
            __m256i cur_idx_vec = _mm256_set_epi32(i + 7, i + 6, i + 5, i + 4, i + 3, i + 2, i + 1, i);

            __m256 cmp_lt = _mm256_cmp_ps(cur_dist_vec, cur_worst_dist_vec, _CMP_GT_OQ);

            cur_worst_dist_vec = _mm256_blendv_ps(cur_worst_dist_vec, cur_dist_vec, cmp_lt);
            cur_worst_idx_vec = _mm256_blendv_epi8(cur_worst_idx_vec, cur_idx_vec, _mm256_castps_si256(cmp_lt));
        }

        // also do the remaining elements
        {
            __m256 cur_dist_vec = _mm256_load_ps(&dist_array[92]);
            __m256i cur_idx_vec = _mm256_set_epi32(99, 98, 97, 96, 95, 94, 93, 92);

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

        for (int i = 0; i < 100; ++i)
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
        worst = 0;
        query_vec = query_vector;
    }

    inline void check(uint32_t vec_idx)
    {
        float dist = dist_to_query(_nodes[vec_idx], *query_vec);
        float worst_dist = dist_array[worst];

        if (fill < 100)
        {
            // insert at the back
            worst = dist < worst_dist ? worst : fill;
            dist_array[fill] = dist;
            vec_idx_array[fill] = vec_idx;
            ++fill;
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

    inline uint32_t size()
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

void vec_query(vector<vector<float>>& nodes, vector<vector<float>>& queries, vector<vector<uint32_t>>& knn_results)
{
    float sample_proportion = 0.001;

    uint32_t n = nodes.size();
    uint32_t d = nodes[0].size();
    uint32_t nq = queries.size();
    uint32_t sn = uint32_t(sample_proportion * n);

    cout << "# data points:  " << n << "\n";
    cout << "# data point dim:  " << d << "\n";
    cout << "# queries:      " << nq << "\n";

    /** A basic method to compute the KNN results using sampling  **/
    const int K = 100;    // To find 100-NN

    Knn knn(nodes);

    for (uint i = 0; i < nq; i++)
    {
        uint32_t query_type = queries[i][0];
        int32_t v = queries[i][1];
        float l = queries[i][2];
        float r = queries[i][3];
        q_vec_t query_vec;

        // first push_back 2 zeros for aligning with dataset
        query_vec.push_back(0);
        query_vec.push_back(0);
        for (uint j = 4; j < queries[i].size(); j++)
            query_vec.push_back(queries[i][j]);

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
        if (knn.size() < K)
        {
            uint32_t s = 1;
            while (knn.size() < K)
            {
                knn.check(n - s);
                s = s + 1;
            }
        }

        knn_results.push_back(knn.get_knn_sorted());
    }
}