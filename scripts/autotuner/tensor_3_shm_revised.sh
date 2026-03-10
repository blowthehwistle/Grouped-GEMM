#!/usr/bin/env bash

set -u

# Define the range of values for each parameter
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
BK_VALUES=(8 16 32 64)
WM_VALUES=(16 32 64 128 256)
WN_VALUES=(16 32 64 128 256)
NUM_THREADS_VALUES=(128)

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

RUNNER="include/runner.cuh"
KERNEL="include/tensor_kernels/3_shm_revised.cuh"
OUTPUT="results/benchmark/tensor_3_shm_revised_8k_t2.txt"
mkdir -p "$(dirname "$OUTPUT")"
#OUTPUT="scirpt_test.txt"

# Clear the output file
echo "" > $OUTPUT

# Set GPU to use
export DEVICE="0"
WARPSIZE=32

TOTAL_CONFIGS="$(( ${#BK_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} * ${#WM_VALUES[@]} * ${#WN_VALUES[@]} * ${#NUM_THREADS_VALUES[@]} ))"
CONFIG_NUM=0

# Loop through all combinations of parameters
for BK in "${BK_VALUES[@]}"; do
	for BM in "${BM_VALUES[@]}"; do
		for BN in "${BN_VALUES[@]}"; do
			for WM in "${WM_VALUES[@]}"; do
				for WN in "${WN_VALUES[@]}"; do
					for NUM_THREADS in "${NUM_THREADS_VALUES[@]}"; do
						echo ""
						CONFIG_NUM=$(( CONFIG_NUM + 1 ))
						# skip configurations that don't fullfil preconditions
						NUM_WARPS=$(( NUM_THREADS / 32 ))
						if ! (( BN % WN == 0 && BM % WM == 0 )); then
							echo "Error: BN % WN must be 0 and BM % WM must be 0."
							continue
						fi
						if ! (( (BN / WN) * (BM / WM) == NUM_WARPS )); then
							echo "Error: (BN / WN) * (BM / WM) must be equal to NUM_WARPS."
							continue
						fi
						if ! (( (NUM_THREADS * 4) % BK == 0 )); then
							echo "Error: (NUM_THREADS * 4) % BK must be 0."
							continue
						fi
						if ! (( (NUM_THREADS * 4) % BN == 0 )); then
							echo "Error: (NUM_THREADS * 4) % BN must be 0."
							continue
						fi
						if ! (( (BM * BK) % (4 * NUM_THREADS) == 0 )); then
							echo "Error: (BM * BK) % (4 * NUM_THREADS) must be 0."
							continue
						fi
						if ! (( (BN * BK) % (4 * NUM_THREADS) == 0 )); then
							echo "Error: (BN * BK) % (4 * NUM_THREADS) must be 0."
							continue
						fi					

						# Update the parameters in the source code
						sed -i "s/const uint t_k3_num_threads = .*/const uint t_k3_num_threads = $NUM_THREADS;/" $RUNNER
						sed -i "s/const uint t_k3_bn = .*/const uint t_k3_bn = $BN;/" $RUNNER
						sed -i "s/const uint t_k3_bm = .*/const uint t_k3_bm = $BM;/" $RUNNER
						sed -i "s/const uint t_k3_bk = .*/const uint t_k3_bk = $BK;/" $RUNNER
						sed -i "s/const uint t_k3_wm = .*/const uint t_k3_wm = $WM;/" $RUNNER
						sed -i "s/const uint t_k3_wn = .*/const uint t_k3_wn = $WN;/" $RUNNER

						# Rebuild the program
						make

						echo "($CONFIG_NUM/$TOTAL_CONFIGS): NUM_THREADS=$NUM_THREADS BM=$BM BN=$BN BK=$BK WM=$WM WN=$WN" |& tee -a $OUTPUT
						# Run the benchmark and get the result
						./bin/tmain_8k 3 0 | tee -a $OUTPUT

						#echo "NUM_THREADS=$NUM_THREADS BM=$BM BN=$BN BK=$BK WM=$WM WN=$WN" |& tee -a $OUTPUT
					done
				done
			done
		done
	done
done