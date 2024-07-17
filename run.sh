mkdir -p "build"
cd "build" || exit
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target clean
cmake --build . -j6

echo ""
echo "Running Baseline Vector Search..."
./baseline.out > ../logs/baseline.txt
echo ""
echo "Running Optimized Vector Search..."
./optimized.out > ../logs/optimized.txt
echo ""
echo "Running Parallel Optimized Vector Search..."
./optimized_parallel.out > ../logs/optimized_parallel.txt
echo ""
echo "Comparing Results"
./compare.out ../result_data/baseline.bin ../result_data/optimized.bin ../result_data/optimized_parallel.bin