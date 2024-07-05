mkdir -p "build"
cd "build" || exit
cmake ..
cmake --build . --target clean
cmake --build .
echo ""
echo "Running Baseline Vector Search..."
start=$(date +%s)
./baseline.out > ../logs/baseline.txt
end=$(date +%s)
echo "Baseline Vector Search took $(($end-$start)) seconds"
echo ""
echo "Running Optimized Vector Search..."
start=$(date +%s)
./optimized.out > ../logs/optimized.txt
end=$(date +%s)
echo "Optimized Vector Search took $(($end-$start)) seconds"
echo ""
echo "Comparing Results"
cmp -l ../result_data/dummy_baseline.bin ../result_data/dummy_optimized.bin
echo "Done"