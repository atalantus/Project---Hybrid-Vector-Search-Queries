#include <iostream>
#include <iomanip>
#include "io.h"

constexpr auto error_delta = 0.002;

template<typename T>
void compare(const std::string& a_path, const std::string& b_path)
{
    std::cout << "\nComparing: " << a_path << " " << b_path << std::endl;

    vector<vector<T>> a_dist_data;
    ReadBinFull(a_path, a_dist_data);

    vector<vector<T>> b_dist_data;
    ReadBinFull(b_path, b_dist_data);

    bool success = true;

    if (a_dist_data.size() != b_dist_data.size())
    {
        std::cerr << "Datasets have different number of queries! " << a_dist_data.size() << ", "
                  << b_dist_data.size() << std::endl;
        return;
    }

    bool same = true;
    uint32_t errs = 0;
    double max_error = 0;

    for (int k = 0; k < a_dist_data.size(); ++k)
    {
        if (a_dist_data[k].size() != b_dist_data[k].size())
        {
            std::cerr << "k query sets have different result sizes!" << a_dist_data[k].size() << ", "
                      << b_dist_data[k].size() << std::endl;
            return;
        }

        for (int l = 0; l < a_dist_data[k].size(); ++l)
        {
            auto diff = std::abs((double) a_dist_data[k][l] - (double) b_dist_data[k][l]);

            if (diff > max_error)
            {
                max_error = diff;
                same = false;
            }

            if (diff >= error_delta)
            {
                success = false;
                errs++;
                if (errs < 50)
                {
                    std::cerr << k << " - " << l << ": distance difference of " << diff << " between "
                              << std::setprecision(15) << a_dist_data[k][l] << " and " << b_dist_data[k][l]
                              << std::endl;
                }
            }
        }
    }

    if (success && same)
    {
        std::cout << "Datasets are the same!" << std::endl;
    } else
    {
        if (success)
        {
            std::cout << "Datasets are similar under error delta!" << std::endl;
        } else
        {
            std::cout << "ERROR: Found a total of " << errs << " differences!" << std::endl;
        }
        std::cout << "Max Floating Point Error Difference: " << std::setprecision(15) << max_error << std::endl;
    }
}

int main(int argc, char* argv[])
{
//    for (int i = 1; i < argc; ++i)
//    {
//        for (int j = 1; j < argc; ++j)
//        {
//            if (i < j)
//            {
//                auto a_path = argv[i];
//                auto b_path = argv[j];
//
//                compare<uint32_t>(a_path, b_path);
//            }
//        }
//    }

    for (int i = 1; i < argc; ++i)
    {
        for (int j = 1; j < argc; ++j)
        {
            if (i < j)
            {
                auto a_path = std::string(argv[i]) + ".dist";
                auto b_path = std::string(argv[j]) + ".dist";

                compare<float>(a_path, b_path);
            }
        }
    }
}