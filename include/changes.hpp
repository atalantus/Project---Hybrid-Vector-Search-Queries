#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <array>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::array;

typedef vector<float> d_vec_t;
typedef vector<float> q_vec_t;

template <typename It, typename T, typename Compare = std::less<>>
auto lower_bound_branchless(It low, It last, const T& val, Compare lt = {}) {
    auto n = std::distance(low, last);

    while (auto half = n / 2) {
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

template<size_t N>
class Knn
{
private:
    static_assert(N > 0);

    const vector<d_vec_t>& _nodes;

    uint32_t fill;
    uint32_t worst;
    q_vec_t* query_vec;

    array<float, N> dist_array;
    array<uint32_t, N> vec_idx_array;

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
        float cur_worst_dist = dist_array[0];
        worst = 0;

        // TODO: SIMD
        for (int i = 0; i < fill; ++i)
        {
            if (dist_array[i] > cur_worst_dist)
            {
                cur_worst_dist = dist_array[i];
                worst = i;
            }
        }
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

        if (fill < N)
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

    Knn<K> knn(nodes);

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