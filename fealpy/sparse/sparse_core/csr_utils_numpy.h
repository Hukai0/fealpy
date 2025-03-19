#ifndef CSR_UTILS_H
#define CSR_UTILS_H

#include <vector>
#include <unordered_map>
// 行索引切片
template<class I, class T>
void csr_row_index(const I n_row_idx, const I rows[], const I Ap[], const I Aj[], const T Ax[], 
                   std::vector<I>& Bj, std::vector<T>& Bx);

// 行范围切片
template<class I, class T>
void csr_row_slice(const I start, const I stop, const I step, const I Ap[], const I Aj[], const T Ax[],
                   std::vector<I>& Bj, std::vector<T>& Bx);

// 列索引切片（Pass 1）
template<class I>
void csr_column_index1(const I n_idx, const I col_idxs[], const I n_row, const I n_col,
                       const I Ap[], const I Aj[], std::vector<I>& col_offsets, std::vector<I>& Bp);

// 列索引切片（Pass 2）
template<class I, class T>
void csr_column_index2(const I col_order[], const I col_offset[], const I nnz,
                       const I Aj[], const T Ax[], std::vector<I>& Bj, std::vector<T>& Bx);

// 根据给定的行列索引提取子矩阵，返回 CSR 格式
template <class I, class T>
void csr_extract_submatrix(const I n_row, const I Ap[], const I Aj[], const T Ax[],
                           const std::vector<I>& rows, const std::vector<I>& cols,
                           std::vector<I>& Bp, std::vector<I>& Bj, std::vector<T>& Bx);

template <typename I, typename T>
void csr_find_nonzero(const I n_row, const I Ap[], const I Aj[], const T Ax[],
                      std::vector<I>& row_indices, std::vector<I>& col_indices, std::vector<T>& values);
     
template <typename I, typename T>
void csr_transpose(const I n_row, const I n_col, const I Ap[], const I Aj[], const T Ax[],
                   std::vector<I>& Bp, std::vector<I>& Bj, std::vector<T>& Bx);

template <typename I, typename T>
void csr_column_min(const I n_row, const I n_col, const I Ap[], const I Aj[], const T Ax[],
                    std::vector<T>& min_values);

#endif // CSR_UTILS_H
