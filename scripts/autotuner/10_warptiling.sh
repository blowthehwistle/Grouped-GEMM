#!/usr/bin/env bash

set -u

# Define the range of values for each parameter
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
BK_VALUES=(8 16 32 64)
WM_VALUES=(32 64 128 256)
WN_VALUES=(32 64 128 256)
WNITER_VALUES=(1 2 4 8)
TM_VALUES=(4 8 16 32)
TN_VALUES=(4 8 16 32)
NUM_THREADS_VALUES=(128)

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

RUNNER="include/runner.cuh"
KERNEL="include/cuda_kernels/10_warptiling.cuh"
OUTPUT="results/benchmark/10_warptiling_4k_t5.txt"
mkdir -p "$(dirname "$OUTPUT")"

# Clear the output file
echo "" > $OUTPUT

# Set GPU to use
export DEVICE="0"
WARPSIZE=32

TOTAL_CONFIGS="$(( ${#BK_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} * ${#WM_VALUES[@]} * ${#WN_VALUES[@]} * ${#WNITER_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#NUM_THREADS_VALUES[@]} ))"
CONFIG_NUM=0

# Loop through all combinations of parameters
for BK in "${BK_VALUES[@]}"; do
	for BM in "${BM_VALUES[@]}"; do
		for BN in "${BN_VALUES[@]}"; do
			for WM in "${WM_VALUES[@]}"; do
				for WN in "${WN_VALUES[@]}"; do
					for WN_ITER in "${WNITER_VALUES[@]}"; do
						for TM in "${TM_VALUES[@]}"; do
							for TN in "${TN_VALUES[@]}"; do
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
									if ! (( (WM * WN) % (WARPSIZE * TM * TN * WN_ITER) == 0 )); then
										echo "Error: (WM * WN) % (WARPSIZE * TM * TN * WN_ITER) must be 0."
										continue
									fi
									WM_ITER=$(( (WM * WN) / (WARPSIZE * TM * TN * WN_ITER) ))
									if ! (( WM % WM_ITER == 0 && WN % WN_ITER == 0 )); then
										echo "Error: WM % WM_ITER must be 0 and WN % WN_ITER must be 0."
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
									if ! (( BN % (16 * TN) == 0 )); then
										echo "Error: BN must be a multiple of 16 * TN."
										continue
									fi
									if ! (( BM % (16 * TM) == 0 )); then
										echo "Error: BM must be a multiple of 16 * TM."
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
									sed -i "s/const uint k10_num_threads = .*/const uint k10_num_threads = $NUM_THREADS;/" $RUNNER
									sed -i "s/const uint k10_bn = .*/const uint k10_bn = $BN;/" $RUNNER
									sed -i "s/const uint k10_bm = .*/const uint k10_bm = $BM;/" $RUNNER
									sed -i "s/const uint k10_bk = .*/const uint k10_bk = $BK;/" $RUNNER
									sed -i "s/const uint k10_wm = .*/const uint k10_wm = $WM;/" $RUNNER
									sed -i "s/const uint k10_wn = .*/const uint k10_wn = $WN;/" $RUNNER
									sed -i "s/const uint k10_wniter = .*/const uint k10_wniter = $WN_ITER;/" $RUNNER
									sed -i "s/const uint k10_tm = .*/const uint k10_tm = $TM;/" $RUNNER
									sed -i "s/const uint k10_tn = .*/const uint k10_tn = $TN;/" $RUNNER

									# Rebuild the program
									make

									echo "($CONFIG_NUM/$TOTAL_CONFIGS): NUM_THREADS=$NUM_THREADS BM=$BM BN=$BN BK=$BK WM=$WM WN=$WN WN_ITER=$WN_ITER TM=$TM TN=$TN" |& tee -a $OUTPUT
									# Run the benchmark and get the result
									timeout -v 10 ./bin/cmain_4k 10 | tee -a $OUTPUT

									echo "NUM_THREADS=$NUM_THREADS BM=$BM BN=$BN BK=$BK WM=$WM WN=$WN WN_ITER=$WN_ITER TM=$TM TN=$TN" |& tee -a $OUTPUT
								done
							done
						done
					done
				done
			done
		done
	done
done