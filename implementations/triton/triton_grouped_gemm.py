# Copyright (c) 2023 - 2025 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
from pathlib import Path
from typing import List, Optional

import torch

import triton
import triton.language as tl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_cuda():
    return torch.cuda.is_available()


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'NUM_SM': num_sms(),
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'NUM_SM': num_sms(),
        }),
        # Small block configs (tl.dot requires all dims >= 16; mask handles partial tiles)
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': num_sms(),
        }),
        triton.Config({
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': num_sms(),
        }),
        triton.Config({
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': num_sms(),
        }),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # workload info for autotune (max(m_list), sum of ceil(m/128)*ceil(n/128) approx)
    max_m,
    total_tiles,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)     # mnk가 3개니까 
        gn = tl.load(group_gemm_sizes + g * 3 + 1)  
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)     #ceiling division
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # Full tiles only (input padded to block multiples) - no mask
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) #현재 타일이 담당하는 M축 인덱스 벡터
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            # 현재 읽어야 하는 타일의 절대주소 계산
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]   # (BLOCK_SIZE_M, 1) 의 열벡터
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]   # (1, BLOCK_SIZE_K) 의 행벡터
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                tl.multiple_of(a_ptrs, [16, 16])    # hint to use vectorized load/
                tl.multiple_of(b_ptrs, [16, 16])
                a = tl.load(a_ptrs)   # 절대주소 보고 GMEM -> Register로 가져오기 / load 와 아래의 연산이 overlap 가능 (Triton이 최적화)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b) # matrix multiplication
                a_ptrs += BLOCK_SIZE_K    
                b_ptrs += BLOCK_SIZE_K * ldb    # 다음 타일로 이동
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]  # c의 절대주소 
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM  #Block swizzling 로직 가능

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


# Pad to block multiples so we never need masks (no partial tiles)
PAD_ALIGN = 128  # max BLOCK_SIZE_M/N/K in configs; all configs divide 128


# triton 예제 코드에서는 mnk가 block size의 배수라고 가정. 
# 여기를 어떻게 처리하느냐에 따라 퍼포먼스가 달라질 것 같다. 
def _pad_grouped_for_triton(
    group_A: List[torch.Tensor],
    group_B: List[torch.Tensor],
    m_list: List[int],
    n_list: List[int],
    k_list: List[int],
):
    """Pad A, B to multiples of PAD_ALIGN so kernel can use mask-free load/store."""
    padded_A, padded_B = [], []
    padded_m, padded_n, padded_k = [], [], []
    for i, (M, N, K) in enumerate(zip(m_list, n_list, k_list)):
        Mp = max(M, 16)
        Np = max(N, 16)
        Kp = max(K, 16)
        Mp = ((Mp + PAD_ALIGN - 1) // PAD_ALIGN) * PAD_ALIGN
        Np = ((Np + PAD_ALIGN - 1) // PAD_ALIGN) * PAD_ALIGN
        Kp = ((Kp + PAD_ALIGN - 1) // PAD_ALIGN) * PAD_ALIGN
        A, B = group_A[i], group_B[i]
        if Mp > M or Np > N or Kp > K:
            Ap = torch.zeros((Mp, Kp), device=A.device, dtype=A.dtype)
            Ap[:M, :K] = A
            Bp = torch.zeros((Kp, Np), device=B.device, dtype=B.dtype)
            Bp[:K, :N] = B
            padded_A.append(Ap)
            padded_B.append(Bp)
            padded_m.append(Mp)
            padded_n.append(Np)
            padded_k.append(Kp)
        else:
            padded_A.append(A)
            padded_B.append(B)
            padded_m.append(M)
            padded_n.append(N)
            padded_k.append(K)
    return padded_A, padded_B, padded_m, padded_n, padded_k


def make_grouped_matrices(
    m_list: List[int],
    n_list: List[int],
    k_list: List[int],
    dtype: torch.dtype = torch.float16,
    device=None,
):
    """M, N, K 리스트로 group_A, group_B 생성."""
    dev = device or DEVICE
    group_A, group_B = [], []
    for M, N, K in zip(m_list, n_list, k_list):
        A = torch.rand((M, K), device=dev, dtype=dtype)
        B = torch.rand((K, N), device=dev, dtype=dtype)
        group_A.append(A)
        group_B.append(B)
    return group_A, group_B


def group_gemm_fn(group_A, group_B):
    assert len(group_A) == len(group_B)
    m_list = [a.shape[0] for a in group_A]
    n_list = [b.shape[1] for b in group_B]
    k_list = [a.shape[1] for a in group_A]

    padded_A, padded_B, padded_m, padded_n, padded_k = _pad_grouped_for_triton(
        group_A, group_B, m_list, n_list, k_list
    )
    group_size = len(padded_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C_pad = []
    for i in range(group_size):
        A, B = padded_A[i], padded_B[i]
        Mp, Np, Kp = padded_m[i], padded_n[i], padded_k[i]
        C = torch.empty((Mp, Np), device=DEVICE, dtype=A.dtype)
        group_C_pad.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [Mp, Np, Kp]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)
    max_m = max(padded_m)
    total_tiles = sum(((m + 127) // 128) * ((n + 127) // 128) for m, n in zip(padded_m, padded_n))
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        d_a_ptrs, d_b_ptrs, d_c_ptrs,
        d_g_sizes, d_g_lds, group_size,
        max_m=max_m, total_tiles=total_tiles,
    )

    return [C[:m, :n].contiguous().clone() for C, m, n in zip(group_C_pad, m_list, n_list)]


tma_configs = [
    triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K' : BK}, num_stages=s, num_warps=w) \
    for BM in [128]\
    for BN in [128, 256]\
    for BK in [64, 128]\
    for s in ([3, 4])\
    for w in [4, 8]\
]


@triton.autotune(
    tma_configs,
    key=['group_a_ptrs', 'group_b_ptrs', 'group_c_ptrs', 'group_size'],
)
@triton.jit
def grouped_matmul_tma_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # is the output FP8 or FP16
    FP8: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8 else tl.float16
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # pick up a tile from the current gemm problem
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(dtype))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(dtype))

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[gm, gk],
                strides=[lda, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            )

            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            )

            # iterate through the tiles in the current gemm problem
            while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
                k = gk
                # figure out tile coordinates
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                # do regular gemm here
                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn = tile_n_idx * BLOCK_SIZE_N

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                    a = a_desc.load([offs_am, kk * BLOCK_SIZE_K])
                    b = b_desc.load([offs_bn, kk * BLOCK_SIZE_K])
                    accumulator += tl.dot(a, b.T)

                offs_cm = tile_m_idx * BLOCK_SIZE_M
                offs_cn = tile_n_idx * BLOCK_SIZE_N

                c = accumulator.to(dtype)
                c_desc.store([offs_cm, offs_cn], c)

                # go to the next tile by advancing NUM_SM
                tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def group_gemm_tma_fn(group_A, group_B):

    assert supports_tma()

    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[1]
        M, K = A.shape
        N, K = B.shape
        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]
    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    # we use a fixed number of CTA, and it's auto-tunable

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_tma_kernel[grid](d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size,
                                    FP8=torch.float8_e4m3fn == group_A[0].dtype, NUM_SM=num_sms())
    return group_C


_DEFAULT_M = [1024, 512, 256, 128]
_DEFAULT_N = [1024, 512, 256, 128]
_DEFAULT_K = [1024, 512, 256, 128]


def _validate_default_shapes():
    """기본 shape으로 Triton vs reference 검증."""
    m, n, k = _DEFAULT_M, _DEFAULT_N, _DEFAULT_K
    group_A = [torch.rand((M, K), device=DEVICE, dtype=torch.float16) for M, K in zip(m, k)]
    group_B = [torch.rand((K, N), device=DEVICE, dtype=torch.float16) for K, N in zip(k, n)]
    tri_out = group_gemm_fn(group_A, group_B)
    ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]
    for i in range(len(m)):
        assert torch.allclose(ref_out[i], tri_out[i], atol=1e-2, rtol=1e-2)
    if supports_tma():
        group_B_T = [b.T.contiguous() for b in group_B]
        tri_tma = group_gemm_tma_fn(group_A, group_B_T)
        for i in range(len(m)):
            assert torch.allclose(ref_out[i], tri_tma[i], atol=1e-2, rtol=1e-2)


# only launch the kernel, no tensor preparation here to remove all overhead
def triton_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size, max_m=0, total_tiles=0):
    if max_m == 0 and total_tiles == 0:
        # fallback: derive from sizes tensor (M,N,K per group: [M0,N0,K0, M1,N1,K1, ...])
        s = sizes.cpu() if sizes.is_cuda else sizes
        sizes_list = s.tolist() if s.dim() == 1 else s.reshape(-1).tolist()
        m_vals = [sizes_list[i * 3] for i in range(group_size)]
        n_vals = [sizes_list[i * 3 + 1] for i in range(group_size)]
        max_m = max(m_vals)
        total_tiles = sum(((m + 127) // 128) * ((n + 127) // 128) for m, n in zip(m_vals, n_vals))
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        a_ptrs, b_ptrs, c_ptrs,
        sizes, lds, group_size,
        max_m=max_m, total_tiles=total_tiles,
    )


def triton_tma_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size, dtype):
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_tma_kernel[grid](a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size, FP8=torch.float8_e4m3fn == dtype,
                                    NUM_SM=num_sms())


def torch_perf_fn(group_A, group_B):
    for a, b in zip(group_A, group_B):
        torch.matmul(a, b)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['N'],
        x_vals=[2**i for i in range(7, 11)],  # different possible values for `x_name`
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['cublas', 'triton'] + (['triton-tma'] if supports_tma() else []),
        # label name for the lines
        line_names=["cuBLAS", "Triton"] + (['Triton + TMA'] if supports_tma() else []),
        # line styles
        styles=[('green', '-'), ('blue', '-')] + ([('red', '-')] if supports_tma() else []),
        ylabel="runtime(ms)",  # label name for the y-axis
        plot_name="group-gemm-performance",
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark_square_matrices(N, provider):
    group_size = 4
    group_A = []
    group_B = []
    group_B_T = []
    A_addrs = []
    B_addrs = []
    B_T_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = torch.rand((N, N), device=DEVICE, dtype=torch.float16)
        B = torch.rand((N, N), device=DEVICE, dtype=torch.float16)
        C = torch.empty((N, N), device=DEVICE, dtype=torch.float16)
        B_T = B.T.contiguous()
        group_A.append(A)
        group_B.append(B)
        group_B_T.append(B_T)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        B_T_addrs.append(B_T.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [N, N, N]
        g_lds += [N, N, N]

    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_b_t_ptrs = torch.tensor(B_T_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    max_m = N
    total_tiles = group_size * ((N + 127) // 128) * ((N + 127) // 128)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_perf_fn(group_A, group_B), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size,
                                  max_m=max_m, total_tiles=total_tiles), quantiles=quantiles)
    if provider == 'triton-tma':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_tma_perf_fn(d_a_ptrs, d_b_t_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size, dtype=torch.
                                       float16), quantiles=quantiles)
    return ms, max_ms, min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['M'],
        x_vals=[2**i for i in range(7, 11)],  # different possible values for `x_name`
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['cublas', 'triton'] + (['triton-tma'] if supports_tma() else []),
        # label name for the lines
        line_names=["cuBLAS", "Triton"] + (['Triton + TMA'] if supports_tma() else []),
        # line styles
        styles=[('green', '-'), ('blue', '-')] + ([('red', '-')] if supports_tma() else []),
        ylabel="runtime(ms)",  # label name for the y-axis
        plot_name="group-gemm-performance-m-8192-k-8192",
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark_batches(M, provider):
    N = 8192
    K = 8192
    group_size = 4
    group_A = []
    group_B = []
    group_B_T = []
    A_addrs = []
    B_addrs = []
    B_T_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    g_T_lds = []
    group_C = []
    for i in range(group_size):
        A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
        B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)
        C = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
        B_T = B.T.contiguous()
        group_A.append(A)
        group_B.append(B)
        group_B_T.append(B_T)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        B_T_addrs.append(B_T.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]
        g_T_lds += [A.stride(0), B_T.stride(0), C.stride(0)]

    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_b_t_ptrs = torch.tensor(B_T_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)
    d_g_t_lds = torch.tensor(g_T_lds, dtype=torch.int32, device=DEVICE)

    max_m = M
    total_tiles = group_size * ((M + 127) // 128) * ((N + 127) // 128)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_perf_fn(group_A, group_B), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size,
                                  max_m=max_m, total_tiles=total_tiles), quantiles=quantiles)
    if provider == 'triton-tma':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_tma_perf_fn(d_a_ptrs, d_b_t_ptrs, d_c_ptrs, d_g_sizes, d_g_t_lds, group_size, dtype=torch.
                                       float16), quantiles=quantiles)
    return ms, max_ms, min_ms


def run_benchmark(
    m_list: List[int],
    n_list: List[int],
    k_list: List[int],
    warmup: int = 50,
    repeat: int = 1000,
) -> tuple:
    """config/CLI로 받은 M,N,K로 벤치마크. (elapsed_ms, gflops) 반환."""
    group_A, group_B = make_grouped_matrices(m_list, n_list, k_list)
    padded_A, padded_B, padded_m, padded_n, padded_k = _pad_grouped_for_triton(
        group_A, group_B, m_list, n_list, k_list
    )
    group_C = [torch.empty((mp, np), device=DEVICE, dtype=torch.float16)
               for mp, np in zip(padded_m, padded_n)]
    A_addrs = [a.data_ptr() for a in padded_A]
    B_addrs = [b.data_ptr() for b in padded_B]
    C_addrs = [c.data_ptr() for c in group_C]
    g_sizes = []
    g_lds = []
    for i in range(len(m_list)):
        Mp, Np, Kp = padded_m[i], padded_n[i], padded_k[i]
        g_sizes += [Mp, Np, Kp]
        g_lds += [padded_A[i].stride(0), padded_B[i].stride(0), group_C[i].stride(0)]
    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)
    max_m = max(padded_m)
    total_tiles = sum(((m + 127) // 128) * ((n + 127) // 128) for m, n in zip(padded_m, padded_n))

    # NVTX range: Nsight Compute에서 --nvtx --nvtx-include "grouped_gemm" 으로
    # 이 구간만 프로파일하면 초기화 커널 없이 실제 GEMM 커널만 수집 가능.
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push("grouped_gemm")
    for _ in range(warmup):
        triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, len(m_list),
                       max_m=max_m, total_tiles=total_tiles)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(repeat):
        triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, len(m_list),
                       max_m=max_m, total_tiles=total_tiles)
    end_ev.record()
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()
    elapsed_ms = start_ev.elapsed_time(end_ev) / repeat
    flops = sum(2 * m * n * k for m, n, k in zip(m_list, n_list, k_list))
    gflops = flops * 1e-9 / (elapsed_ms / 1000)
    return elapsed_ms, gflops


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "experiments" / "compare_grouped"))
    try:
        from script_utils import load_grouped_config_module, DEFAULT_CONFIG
        _load, _add_args = load_grouped_config_module()
    except ImportError:
        _load = _add_args = None
        DEFAULT_CONFIG = None

    parser = argparse.ArgumentParser()
    if _add_args:
        _add_args(parser)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--no-config", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true", help="config 모드로만 실행, perf_report 스킵")
    args = parser.parse_args()

    use_config = args.benchmark_only or args.config or args.m or args.n or args.k
    if use_config and _load and DEFAULT_CONFIG:
        m_list, n_list, k_list = _load(
            config_path=None if args.no_config else (args.config or DEFAULT_CONFIG),
            m=args.m, n=args.n, k=args.k,
            default_config=DEFAULT_CONFIG,
        )
        # Validation: Triton output vs torch.matmul reference
        group_A, group_B = make_grouped_matrices(m_list, n_list, k_list, dtype=torch.float16)
        padded_A, padded_B, padded_m, padded_n, padded_k = _pad_grouped_for_triton(
            group_A, group_B, m_list, n_list, k_list
        )
        group_C = [torch.empty((mp, np), device=DEVICE, dtype=torch.float16)
                   for mp, np in zip(padded_m, padded_n)]
        A_addrs = [a.data_ptr() for a in padded_A]
        B_addrs = [b.data_ptr() for b in padded_B]
        C_addrs = [c.data_ptr() for c in group_C]
        g_sizes = []
        g_lds = []
        for i in range(len(m_list)):
            Mp, Np, Kp = padded_m[i], padded_n[i], padded_k[i]
            g_sizes += [Mp, Np, Kp]
            g_lds += [padded_A[i].stride(0), padded_B[i].stride(0), group_C[i].stride(0)]
        d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
        d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
        d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
        d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
        d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)
        max_m = max(padded_m)
        total_tiles = sum(((m + 127) // 128) * ((n + 127) // 128) for m, n in zip(padded_m, padded_n))
        triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, len(m_list),
                       max_m=max_m, total_tiles=total_tiles)
        torch.cuda.synchronize()
        ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]
        for i in range(len(m_list)):
            m, n = m_list[i], n_list[i]
            assert torch.allclose(
                group_C[i][:m, :n].float(), ref_out[i].float(),
                atol=1e-1, rtol=1e-1,
            ), f"Triton validation failed at batch {i}"
        print("Triton grouped GEMM validation passed.")

        elapsed_ms, gflops = run_benchmark(m_list, n_list, k_list, args.warmup, args.repeat)
        print(f"M,N,K: {m_list}, {n_list}, {k_list}")
        print(f"Triton grouped GEMM: {elapsed_ms:.3f} ms, {gflops:.1f} GFLOPS")
    else:
        _validate_default_shapes()
        benchmark_square_matrices.run(show_plots=True, print_data=True)
        benchmark_batches.run(show_plots=True, print_data=True)
