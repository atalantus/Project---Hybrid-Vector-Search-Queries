#include "io.h"
#include "perfevent.hpp"

#define DATASET 0

#if IMPL == 2
#include "optimized.hpp"
#elif IMPL == 3
#include "optimized_parallel.hpp"
#else
#include "baseline.hpp"

#endif

using std::cout;
using std::endl;
using std::string;
using std::vector;

int main(int argc, char** argv)
{
#if DATASET == 1
    const std::string dataset = "medium";
#elif DATASET == 2
    const std::string dataset = "large";
#else
    const std::string dataset = "default";
#endif

#if IMPL == 2
    std::cout << "Running Optimized Vector Search\n";
#elif IMPL == 3
    std::cout << "Running Parallel Optimized Vector Search\n";
#else
    std::cout << "Running Baseline Vector Search\n";
#endif


    string source_path = "../data/" + dataset + "-data.bin";
    string query_path = "../data/query.bin";

#if IMPL == 2
    string knn_save_path = "../result_data/optimized.bin";
#elif IMPL == 3
    string knn_save_path = "../result_data/optimized_parallel.bin";
#else
    string knn_save_path = "../result_data/baseline.bin";
#endif

    // Also accept other path for source data
    switch (argc)
    {
        case 4:
            knn_save_path = string(argv[3]);
        case 3:
            query_path = string(argv[2]);
        case 2:
            source_path = string(argv[1]);
        case 1:
            break;
        default:
            cout << std::string(argv[0]) << " [source_path] [query_path] [output_path]\n";
            exit(1);
    }

    uint32_t num_data_dimensions = 102;

    float sample_proportion = /*0.001*/1;

    // Read data points
    vector<vector<float>> nodes;
    ReadBin(source_path, num_data_dimensions, nodes);
    cout << nodes.size() << "\n";

    // Read queries
    uint32_t num_query_dimensions = num_data_dimensions + 2;
    vector<vector<float>> queries;
    ReadBin(query_path, num_query_dimensions, queries);

    vector<vector<uint32_t>> knn_results;

    PerfEvent e;
    e.startCounters();

    vec_query(nodes, queries, sample_proportion, knn_results);

    e.stopCounters();
    e.printReport(std::cout, 1000);
    std::cout << std::endl;

    std::cerr << "Vector Search took " << e.getDurationInNs() / 1e6 << " ms @ " << std::setprecision(3) << e.getGHz()
              << " GHz" << std::endl;

    // save the results
    SaveKNN(knn_results, knn_save_path);

    vector<vector<vector<float>>> fullKnn;
    for (auto& knn_result: knn_results)
    {
        vector<vector<float>> queryKnns;

        for (int j = 0; j < 100; ++j)
        {
            queryKnns.push_back(nodes[knn_result[j]]);
        }

        fullKnn.push_back(queryKnns);
    }

    SaveKNNFull(fullKnn, queries, std::string(knn_save_path) + ".dist");
    return 0;
}