/**
 * Grouped GEMM: cuBLAS Grouped API only.
 * Single program, no baseline; ncu can profile from the first launches.
 */
#include "grouped_gemm_common.cuh"

int main(int argc, char **argv) {
  GroupedGemmParams *p = grouped_gemm_setup(argc, argv);
  if (!p) return EXIT_FAILURE;

  for (int i = 0; i < GROUPED_GEMM_WARMUP; i++)
    runCublasGroupedTF32_with_TC(p->handle, p->d_A, p->d_B, p->d_C, p->m_list, p->n_list, p->k_list,
                                 static_cast<int>(p->batch_size), p->A_offset, p->B_offset,
                                 p->C_offset, p->alpha, p->beta);
  cudaDeviceSynchronize();

  run_ref_loop(p);
  cudaDeviceSynchronize();

  nvtxRangePushA("grouped_gemm");
  CHECK_CUDA(cudaEventRecord(p->start));
  for (int i = 0; i < p->repeat_time; i++)
    runCublasGroupedTF32_with_TC(p->handle, p->d_A, p->d_B, p->d_C, p->m_list, p->n_list, p->k_list,
                                 static_cast<int>(p->batch_size), p->A_offset, p->B_offset,
                                 p->C_offset, p->alpha, p->beta);
  CHECK_CUDA(cudaEventRecord(p->stop));
  CHECK_CUDA(cudaEventSynchronize(p->stop));
  nvtxRangePop();

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, p->start, p->stop));
  CHECK_CUDA(cudaMemcpy(p->C, p->d_C, p->size_C * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(p->C_ref, p->d_C_ref, p->size_C_ref * sizeof(float), cudaMemcpyDeviceToHost));

  grouped_gemm_verify_and_print(p, elapsed_ms);
  grouped_gemm_teardown(p);
  return 0;
}
