#pragma once

#include "cuda_kernels/1_naive.cuh"
#include "cuda_kernels/2_global_coalesce.cuh"
#include "cuda_kernels/3_shared_caching.cuh"
#include "cuda_kernels/4_blocking_1d.cuh"
#include "cuda_kernels/5_blocking_2d.cuh"
#include "cuda_kernels/6_vectorized.cuh"
#include "cuda_kernels/6_vectorized_revised.cuh"
#include "cuda_kernels/7_resolve_bank_conflict.cuh"
#include "cuda_kernels/10_warptiling.cuh"
#include "cuda_kernels/11_double_buffering.cuh"