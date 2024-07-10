mkdir -p "build"
cd "build" || exit
cmake ..
cmake --build . --target clean
cmake --build .

echo ""
echo "Running Baseline Vector Search..."
./baseline.out > ../logs/baseline.txt
echo ""
echo "Running Optimized Vector Search..."
./optimized.out > ../logs/optimized.txt
echo ""
echo "Running Changed Vector Search..."
./changes.out > ../logs/changes.txt
echo ""
echo "Comparing Results"
cmp ../result_data/dummy_baseline.bin ../result_data/dummy_optimized.bin
cmp ../result_data/dummy_baseline.bin ../result_data/dummy_changes.bin
echo "Done"