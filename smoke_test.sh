echo "=== CPU + PollardTests ==="
make clean
make pollard-tests
bin/pollardtests || exit 1

echo "=== CUDA build & tests ==="
make clean
make BUILD_CUDA=1
make pollard-tests
bin/pollardtests || exit 2

echo "=== OpenCL build & tests ==="
make clean
make BUILD_OPENCL=1
make pollard-tests
bin/pollardtests || exit 3

echo "All tests passed—BitCrack now matches the Python behavior!"
