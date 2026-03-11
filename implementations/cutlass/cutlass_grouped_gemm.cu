/**
 * Cutlass Grouped GEMM - 우리 코드베이스와 align
 *
 * - Config: grouped_config.txt (한 줄당 "M N K", CUDA/torch/triton과 동일)
 * - CLI: ./bin/tmain_cutlass_grouped <config_file> [p|v]
 *   p=compact (compare.py 파싱용), v=verbose
 * - Output: CUDA와 동일 형식 (Batch size, Shapes, B M N K, Result, Performance, [raw])
 *
 * CUTLASS 헤더 필요. 빌드: make cutlass_grouped (CUTLASS_ROOT 설정)
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/device/gemm.h"

#include "helpers.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Config 파일에서 M,N,K 로드 (한 줄당 "M N K"). CUDA grouped_gemm.cu와 동일 형식.
int load_mnk_from_file(const char *path, std::vector<int> &m_vec, std::vector<int> &n_vec,
                       std::vector<int> &k_vec) {
  std::ifstream f(path);
  if (!f) return -1;
  m_vec.clear();
  n_vec.clear();
  k_vec.clear();
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    int a, b, c;
    if (ss >> a >> b >> c) {
      m_vec.push_back(a);
      n_vec.push_back(b);
      k_vec.push_back(c);
    }
  }
  return m_vec.empty() ? -1 : static_cast<int>(m_vec.size());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::ColumnMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, 8,
    ElementOutput, LayoutC, ElementAccumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 4>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
using GemmCoord = cutlass::gemm::GemmCoord;

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <config_file> [p|v]\n"
              << "  config_file: 한 줄당 'M N K' (CUDA grouped_config.txt와 동일)\n"
              << "  p=compact, v=verbose\n";
    return EXIT_FAILURE;
  }

  cudaDeviceProp props;
  if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties failed." << std::endl;
    return EXIT_FAILURE;
  }
  if (__CUDACC_VER_MAJOR__ < 11 || props.major < 8) {
    std::cerr << "CUTLASS Grouped GEMM requires Ampere (sm80) or later." << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<int> m_vec, n_vec, k_vec;
  int batch_size = load_mnk_from_file(argv[1], m_vec, n_vec, k_vec);
  if (batch_size <= 0) {
    std::cerr << "Failed to load config from " << argv[1] << std::endl;
    return EXIT_FAILURE;
  }

  bool compact = (argc >= 3 && argv[2][0] == 'p');

  // problem_sizes
  std::vector<GemmCoord> problem_sizes(batch_size);
  for (int i = 0; i < batch_size; i++) {
    problem_sizes[i] = GemmCoord(m_vec[i], n_vec[i], k_vec[i]);
  }

  // offsets, lda/ldb/ldc/ldd
  std::vector<int64_t> offset_A(batch_size + 1), offset_B(batch_size + 1),
      offset_C(batch_size + 1), offset_D(batch_size + 1);
  std::vector<int64_t> lda_host(batch_size), ldb_host(batch_size),
      ldc_host(batch_size), ldd_host(batch_size);

  int64_t total_A = 0, total_B = 0, total_C = 0, total_D = 0;
  for (int i = 0; i < batch_size; i++) {
    offset_A[i] = total_A;
    offset_B[i] = total_B;
    offset_C[i] = total_C;
    offset_D[i] = total_D;
    int64_t M = m_vec[i], N = n_vec[i], K = k_vec[i];
    lda_host[i] = M;  // column-major A(M,K) -> stride M
    ldb_host[i] = K;  // column-major B(K,N) -> stride K
    ldc_host[i] = M;  // column-major C(M,N) -> stride M
    ldd_host[i] = M;
    total_A += M * K;
    total_B += K * N;
    total_C += M * N;
    total_D += M * N;
  }
  offset_A[batch_size] = total_A;
  offset_B[batch_size] = total_B;
  offset_C[batch_size] = total_C;
  offset_D[batch_size] = total_D;

  // Device allocations
  cutlass::DeviceAllocation<GemmCoord> problem_sizes_device(batch_size);
  cutlass::DeviceAllocation<int64_t> lda(batch_size), ldb(batch_size), ldc(batch_size), ldd(batch_size);
  cutlass::DeviceAllocation<ElementA> block_A(total_A);
  cutlass::DeviceAllocation<ElementB> block_B(total_B);
  cutlass::DeviceAllocation<ElementOutput> block_C(total_C);
  cutlass::DeviceAllocation<ElementOutput> block_D(total_D);
  cutlass::DeviceAllocation<ElementOutput> block_Ref(total_D);
  cutlass::DeviceAllocation<ElementA *> ptr_A(batch_size);
  cutlass::DeviceAllocation<ElementB *> ptr_B(batch_size);
  cutlass::DeviceAllocation<ElementOutput *> ptr_C(batch_size);
  cutlass::DeviceAllocation<ElementOutput *> ptr_D(batch_size);

  problem_sizes_device.copy_from_host(problem_sizes.data());
  lda.copy_from_host(lda_host.data());
  ldb.copy_from_host(ldb_host.data());
  ldc.copy_from_host(ldc_host.data());
  ldd.copy_from_host(ldd_host.data());

  std::vector<ElementA *> ptr_A_host(batch_size);
  std::vector<ElementB *> ptr_B_host(batch_size);
  std::vector<ElementOutput *> ptr_C_host(batch_size);
  std::vector<ElementOutput *> ptr_D_host(batch_size);
  for (int i = 0; i < batch_size; i++) {
    ptr_A_host[i] = block_A.get() + offset_A[i];
    ptr_B_host[i] = block_B.get() + offset_B[i];
    ptr_C_host[i] = block_C.get() + offset_C[i];
    ptr_D_host[i] = block_D.get() + offset_D[i];
  }
  ptr_A.copy_from_host(ptr_A_host.data());
  ptr_B.copy_from_host(ptr_B_host.data());
  ptr_C.copy_from_host(ptr_C_host.data());
  ptr_D.copy_from_host(ptr_D_host.data());

  cutlass::reference::device::BlockFillRandomUniform(
      block_A.get(), total_A, 3080u, ElementA(8), ElementA(-8), 0);
  cutlass::reference::device::BlockFillRandomUniform(
      block_B.get(), total_B, 3081u, ElementB(8), ElementB(-8), 0);
  cutlass::reference::device::BlockFillRandomUniform(
      block_C.get(), total_C, 3082u, ElementOutput(2), ElementOutput(-2), 0);
  cutlass::reference::device::BlockFillSequential(
      block_D.get(), total_D, ElementOutput(), ElementOutput());

  // Print config (CUDA와 동일 형식)
  std::cout << "=== Cutlass Grouped GEMM Benchmark ===\n";
  std::cout << "Batch size: " << batch_size << "\n";
  std::cout << "Shapes: [";
  for (int i = 0; i < batch_size; i++) {
    if (i) std::cout << ", ";
    std::cout << "(" << m_vec[i] << "," << n_vec[i] << "," << k_vec[i] << ")";
  }
  std::cout << "]\n\n";
  std::cout << "      B          M     N     K\n";
  int check_point = 0;
  for (int batch = 1; batch < batch_size; batch++) {
    if (m_vec[batch] != m_vec[check_point] || n_vec[batch] != n_vec[check_point] ||
        k_vec[batch] != k_vec[check_point]) {
      std::printf("%6d-%-5d%6d%6d%6d\n", check_point, batch - 1, m_vec[batch - 1], n_vec[batch - 1],
                  k_vec[batch - 1]);
      check_point = batch;
    }
  }
  std::printf("%6d-%-5d%6d%6d%6d\n", check_point, batch_size - 1, m_vec[check_point],
              n_vec[check_point], k_vec[check_point]);

  int threadblock_count = GemmGrouped::sufficient(problem_sizes.data(), batch_size);
  if (!threadblock_count) {
    std::cerr << "CUTLASS Grouped GEMM: insufficient hardware resources." << std::endl;
    return EXIT_FAILURE;
  }

  float alpha = 1.0f, beta = 0.0f;
  typename GemmGrouped::EpilogueOutputOp::Params epilogue_op(alpha, beta);

  typename GemmGrouped::Arguments args(
      problem_sizes_device.get(), batch_size, threadblock_count, epilogue_op, ptr_A.get(),
      ptr_B.get(), ptr_C.get(), ptr_D.get(), lda.get(), ldb.get(), ldc.get(), ldd.get(),
      problem_sizes.data());

  GemmGrouped gemm;
  size_t workspace_size = gemm.get_workspace_size(args);
  cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) {
    std::cerr << "Failed to initialize CUTLASS Grouped GEMM." << std::endl;
    return EXIT_FAILURE;
  }

  // Run once to get output
  gemm.run();

  // Compute reference: for each batch, ref = alpha * A*B + beta * C
  for (int i = 0; i < batch_size; i++) {
    int M = m_vec[i], N = n_vec[i], K = k_vec[i];
    cutlass::gemm::GemmCoord prob(M, N, K);
    LayoutA layout_a(lda_host[i]);
    LayoutB layout_b(ldb_host[i]);
    LayoutC layout_c(ldc_host[i]);
    cutlass::TensorRef<ElementA, LayoutA> ref_a(block_A.get() + offset_A[i], layout_a);
    cutlass::TensorRef<ElementB, LayoutB> ref_b(block_B.get() + offset_B[i], layout_b);
    cutlass::TensorRef<ElementOutput, LayoutC> ref_c(block_C.get() + offset_C[i], layout_c);
    cutlass::TensorRef<ElementOutput, LayoutC> ref_d(block_Ref.get() + offset_D[i], layout_c);
    cutlass::reference::device::compute_gemm<
        ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
        float, ElementAccumulator>(
        prob, alpha, ref_a, ref_b, beta, ref_c, ref_d, ElementAccumulator(0));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy to host and compare
  std::vector<ElementOutput> host_D(total_D), host_Ref(total_D);
  CHECK_CUDA(cudaMemcpy(host_D.data(), block_D.get(), total_D * sizeof(ElementOutput),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_Ref.data(), block_Ref.get(), total_D * sizeof(ElementOutput),
                        cudaMemcpyDeviceToHost));

  using MatrixCoord = cutlass::MatrixCoord;
  bool passed = true;
  ElementOutput epsilon(0.1f);   // FP16 tolerance
  ElementOutput floor(1e-3f);
  for (int i = 0; i < batch_size; i++) {
    int M = m_vec[i], N = n_vec[i];
    cutlass::TensorView<ElementOutput, LayoutC> view_D(
        host_D.data() + offset_D[i], LayoutC(M), MatrixCoord(M, N));
    cutlass::TensorView<ElementOutput, LayoutC> view_Ref(
        host_Ref.data() + offset_D[i], LayoutC(M), MatrixCoord(M, N));
    if (!cutlass::reference::host::TensorRelativelyEquals(view_D, view_Ref, epsilon, floor)) {
      passed = false;
      break;
    }
  }

  if (passed) {
    std::cout << "Result is correct\n";
  } else {
    std::cout << "Result is different\n";
    return EXIT_FAILURE;
  }

  // Warmup
  gemm.run();

  int repeat = 1000;
  cudaEvent_t start_ev, stop_ev;
  CHECK_CUDA(cudaEventCreate(&start_ev));
  CHECK_CUDA(cudaEventCreate(&stop_ev));

  CHECK_CUDA(cudaEventRecord(start_ev));
  for (int i = 0; i < repeat; i++) {
    gemm.run();
  }
  CHECK_CUDA(cudaEventRecord(stop_ev));
  CHECK_CUDA(cudaEventSynchronize(stop_ev));

  float elapsed_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev));

  int64_t flops = 0;
  for (int i = 0; i < batch_size; i++) {
    flops += 2LL * m_vec[i] * n_vec[i] * k_vec[i];
  }
  float t_per_iter_ms = elapsed_ms / repeat;
  float t_per_iter_s = t_per_iter_ms / 1000.f;
  float gflops = (repeat * flops * 1e-9f) / (elapsed_ms / 1000.f);

  if (compact) {
    std::cout << "\n--- Performance ---\n";
    std::cout << "  Cutlass Grouped:      " << t_per_iter_ms << " ms/iter,  " << gflops
              << " GFLOPS\n";
    std::cout << "Cutlass grouped GEMM: " << t_per_iter_ms << " ms, " << gflops << " GFLOPS\n";
    std::cout << "  [raw] 0," << t_per_iter_s << ",0," << gflops << "\n";
  } else {
    std::cout << "\nCutlass Grouped Runtime: " << t_per_iter_ms << " ms/iter\n";
    std::cout << "Cutlass Grouped GFLOPS:  " << gflops << "\n";
  }

  CHECK_CUDA(cudaEventDestroy(start_ev));
  CHECK_CUDA(cudaEventDestroy(stop_ev));

  return 0;
}
