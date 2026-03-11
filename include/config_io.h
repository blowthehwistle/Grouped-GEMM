/**
 * Grouped GEMM config I/O - shared by grouped_gemm.cu, grouped_gemm_half.cu
 * 한 줄당 "M N K" 형식의 config 파일 로드.
 */
#pragma once

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

/**
 * config 파일에서 M,N,K 로드. 한 줄당 "M N K" (배치 1개).
 * 반환: batch_size, 또는 -1 (오류)
 * *m_list, *n_list, *k_list에 malloc으로 할당 (caller가 free)
 */
inline int load_mnk_from_file(const char *path, int **m_list, int **n_list, int **k_list) {
  std::ifstream f(path);
  if (!f) return -1;
  std::vector<int> m_vec, n_vec, k_vec;
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
  if (m_vec.empty()) return -1;
  int batch_size = static_cast<int>(m_vec.size());
  *m_list = (int *)malloc(sizeof(int) * (batch_size + 1));
  *n_list = (int *)malloc(sizeof(int) * (batch_size + 1));
  *k_list = (int *)malloc(sizeof(int) * (batch_size + 1));
  for (int i = 0; i < batch_size; i++) {
    (*m_list)[i] = m_vec[i];
    (*n_list)[i] = n_vec[i];
    (*k_list)[i] = k_vec[i];
  }
  return batch_size;
}
