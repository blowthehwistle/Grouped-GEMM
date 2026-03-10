SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

PREFIX="large"

# Target directory name
TARGET_DIR="profiles/nsight"

# Check if the directory exists; if not, create it
if [ ! -d "$TARGET_DIR" ]; then
    echo "Directory [$TARGET_DIR] does not exist. Creating it..."
    mkdir -p "$TARGET_DIR"
else
    echo "Directory [$TARGET_DIR] already exists."
fi

make clean
make

# Nsight Compute
# ncu --launch-count 40 --nvtx --nvtx-include "Kernel/" -o profiles/nsight/${PREFIX}_ncu_cublas_iterative -f --set full ./bin/tmain_grouped 0 p
# ncu --launch-count 5 --nvtx --nvtx-include "Kernel/" -o profiles/nsight/${PREFIX}_ncu_cublas_grouped -f --set full ./bin/tmain_grouped 1 p
# ncu --launch-count 5 --nvtx --nvtx-include "Kernel/" --export "profiles/nsight/profile_$(date +%Y%m%d_%H%M%S).ncu-rep" -f --set full ./bin/tmain_grouped 2 p
# ncu --launch-count 40 --nvtx --nvtx-include "Kernel/" -o profiles/nsight/${PREFIX}_ncu_cublas_iterative_half -f --set full ./bin/tmain_grouped_half 0 p
# ncu --launch-count 5 --nvtx --nvtx-include "Kernel/" -o profiles/nsight/${PREFIX}_ncu_cublas_grouped_half -f --set full ./bin/tmain_grouped_half 1 p

ncu --launch-count 5 --nvtx --nvtx-include "Kernel/" --export "profiles/nsight/profile_$(date +%Y%m%d_%H%M%S).ncu-rep" -f --set full ./bin/tmain_grouped 0 p
ncu --launch-count 5 --nvtx --nvtx-include "Kernel/" --export "profiles/nsight/profile_$(date +%Y%m%d_%H%M%S).ncu-rep" -f --set full ./bin/tmain_grouped 1 p

# Nsight Profile
# nsys profile -t cuda,cublas,cusparse,nvtx --cudabacktrace=all -o profiles/nsight/${PREFIX}_nsys_cublas_grouped ./bin/tmain_grouped 1 p
# nsys profile -t cuda,cublas,cusparse,nvtx --cudabacktrace=all -o profiles/nsight/${PREFIX}_nsys_custom_kernel ./bin/tmain_grouped 2 p
# nsys profile -t cuda,cublas,cusparse,nvtx --cudabacktrace=all -o profiles/nsight/${PREFIX}_nsys_cublas_grouped_half ./bin/tmain_grouped_half 1 p