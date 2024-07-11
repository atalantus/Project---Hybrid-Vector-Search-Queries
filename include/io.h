/**
 *  Example code for IO, read binary data vectors and save KNNs to path.
 *
 */

#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "assert.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knn a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void SaveKNN(const std::vector<std::vector<uint32_t>>& knns,
             const std::string& path = "output.bin")
{
    std::ofstream ofs(path, std::ios::out | std::ios::binary);
    const int K = 100;
    const uint32_t N = knns.size();
    assert(knns.front().size() == K);
    for (unsigned i = 0; i < N; ++i)
    {
        auto const& knn = knns[i];
        ofs.write(reinterpret_cast<char const*>(&knn[0]), K * sizeof(uint32_t));
    }
    ofs.close();
}

float calc_dist(const std::vector<float>& a, const std::vector<float>& b)
{
    float sum = 0.0;
    // Skip the first 2 dimensions
    for (size_t i = 2; i < a.size(); ++i)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

void SaveKNNFull(const std::vector<std::vector<std::vector<float>>>& knns, std::vector<std::vector<float>> query_vecs,
                 const std::string& path = "output.bin.dist")
{
    std::ofstream ofs(path, std::ios::out | std::ios::binary);
    const int K = 100;
    const uint32_t N = knns.size();
    assert(knns.front().size() == K);

    ofs.write(reinterpret_cast<char const*>(&N), sizeof(uint32_t));

    for (unsigned i = 0; i < N; ++i)
    {
        for (int l = 0; l < K; ++l)
        {
            vector<float> query_vec;

            // first push_back 2 zeros for aligning with dataset
            query_vec.push_back(0);
            query_vec.push_back(0);
            for (uint j = 4; j < query_vecs[i].size(); j++)
                query_vec.push_back(query_vecs[i][j]);

            auto dist = calc_dist(knns[i][l], query_vec);

            ofs.write(reinterpret_cast<char const*>(&dist), sizeof(float));
        }
    }
    ofs.close();
}

template<typename T>
void ReadBinFull(const std::string& file_path,
                 std::vector<std::vector<T>>& data)
{
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    uint32_t N;  // num of query results
    ifs.read((char*) &N, sizeof(uint32_t));
    data.resize(N);

    for (int i = 0; i < N; ++i)
    {
        data[i].resize(100);
        for (int j = 0; j < 100; ++j)
        {
            std::vector<T> buff(1);
            ifs.read((char*) buff.data(), sizeof(T));

            auto f = static_cast<T>(buff[0]);
            data[i][j] = f;
        }
    }

    ifs.close();
}


/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadBin(const std::string& file_path,
             const int num_dimensions,
             std::vector<std::vector<float>>& data)
{
    std::cout << "Reading Data: " << file_path << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    uint32_t N;  // num of points
    ifs.read((char*) &N, sizeof(uint32_t));
    data.resize(N);
    std::cout << "# of points: " << N << std::endl;
    std::vector<float> buff(num_dimensions);
    int counter = 0;
    while (ifs.read((char*) buff.data(), num_dimensions * sizeof(float)))
    {
        std::vector<float> row(num_dimensions);
        for (int d = 0; d < num_dimensions; d++)
        {
            row[d] = static_cast<float>(buff[d]);
        }
        data[counter++] = std::move(row);
    }
    ifs.close();
    std::cout << "Finish Reading Data" << endl;
}