mkdir -p "build"
cd "build" || exit
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target clean
cmake --build .

echo ""
echo "Running Baseline Vector Search..."
./baseline.out > ../logs/baseline.txt
echo ""
echo "Running Optimized_1 Vector Search..."
./optimized_1.out > ../logs/optimized_1.txt
echo ""
echo "Running Optimized_2 Vector Search..."
./optimized_2.out > ../logs/optimized_2.txt
echo ""
echo "Comparing Results"
./compare.out ../result_data/dummy_baseline.bin ../result_data/dummy_optimized_1.bin ../result_data/dummy_optimized_2.bin