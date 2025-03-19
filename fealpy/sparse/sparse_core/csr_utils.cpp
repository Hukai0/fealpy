#include "csr_utils.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <unordered_map>
// 行索引切片
template<class I, class T>
void csr_row_index(const I n_row_idx, const I rows[], const I Ap[], const I Aj[], const T Ax[], 
                   std::vector<I>& Bj, std::vector<T>& Bx) {
    // 预分配目标向量的大小
    Bj.reserve(n_row_idx * (Ap[rows[0] + 1] - Ap[rows[0]]));  // 假设所有行的非零元素个数差异不大
    Bx.reserve(n_row_idx * (Ap[rows[0] + 1] - Ap[rows[0]]));
    
    for (I i = 0; i < n_row_idx; i++) {
        const I row = rows[i];
        const I row_start = Ap[row];
        const I row_end = Ap[row + 1];
        Bj.insert(Bj.end(), Aj + row_start, Aj + row_end);
        Bx.insert(Bx.end(), Ax + row_start, Ax + row_end);
    }
}

// 行范围切片
template<class I, class T>
void csr_row_slice(const I start, const I stop, const I step, const I Ap[], const I Aj[], const T Ax[],
                   std::vector<I>& Bj, std::vector<T>& Bx) {
    // 预分配空间
    Bj.reserve((stop - start) * (Ap[start + 1] - Ap[start]));
    Bx.reserve((stop - start) * (Ap[start + 1] - Ap[start]));

    if (step > 0) {
        for(I row = start; row < stop; row += step) {
            const I row_start = Ap[row];
            const I row_end = Ap[row + 1];
            Bj.insert(Bj.end(), Aj + row_start, Aj + row_end);
            Bx.insert(Bx.end(), Ax + row_start, Ax + row_end);
        }
    } else {
        for(I row = start; row > stop; row += step) {
            const I row_start = Ap[row];
            const I row_end = Ap[row + 1];
            Bj.insert(Bj.end(), Aj + row_start, Aj + row_end);
            Bx.insert(Bx.end(), Ax + row_start, Ax + row_end);
        }
    }
}

// 列索引切片（Pass 1）
template<class I>
void csr_column_index1(const I n_idx, const I col_idxs[], const I n_row, const I n_col,
                       const I Ap[], const I Aj[], std::vector<I>& col_offsets, std::vector<I>& Bp) {
    // 初始化列偏移数组
    col_offsets.assign(n_col, 0);
    for(I jj = 0; jj < n_idx; jj++) {
        col_offsets[col_idxs[jj]]++;
    }

    I new_nnz = 0;
    Bp.resize(n_row + 1);
    Bp[0] = 0;
    for(I i = 0; i < n_row; i++) {
        for(I jj = Ap[i]; jj < Ap[i + 1]; jj++) {
            new_nnz += col_offsets[Aj[jj]];
        }
        Bp[i + 1] = new_nnz;
    }

    // 更新列偏移量
    for(I j = 1; j < n_col; j++) {
        col_offsets[j] += col_offsets[j - 1];
    }
}

// 列索引切片（Pass 2）
template<class I, class T>
void csr_column_index2(const I col_order[], const I col_offset[], const I nnz,
                       const I Aj[], const T Ax[], std::vector<I>& Bj, std::vector<T>& Bx) {
    // 预分配空间
    Bj.resize(nnz);
    Bx.resize(nnz);

    I n = 0;
    for(I jj = 0; jj < nnz; jj++) {
        const I j = Aj[jj];
        const I offset = col_offset[j];
        const I prev_offset = (j == 0) ? 0 : col_offset[j - 1];
        if (offset != prev_offset) {
            const T v = Ax[jj];
            for(I k = prev_offset; k < offset; k++) {
                Bj[n] = col_order[k];
                Bx[n] = v;
                n++;
            }
        }
    }
}


// 根据给定的行列索引提取子矩阵，返回 CSR 格式
template <class I, class T>
void csr_extract_submatrix(const I n_row, const I Ap[], const I Aj[], const T Ax[],
                           const std::vector<I>& rows, const std::vector<I>& cols,
                           std::vector<I>& Bp, std::vector<I>& Bj, std::vector<T>& Bx) {
    Bp.clear();
    Bj.clear();
    Bx.clear();
    
    I new_n_row = rows.size();
    Bp.resize(new_n_row + 1, 0);
    
    std::unordered_map<I, I> col_map;
    for (I i = 0; i < static_cast<I>(cols.size()); i++) {

        col_map[cols[i]] = i;  // 重新编号列索引
    }

    I nnz = 0;  // 新矩阵的非零元个数
    for (I i = 0; i < new_n_row; i++) {
        I row = rows[i];
        I row_start = Ap[row];
        I row_end = Ap[row + 1];

        for (I jj = row_start; jj < row_end; jj++) {
            I col = Aj[jj];
            if (col_map.find(col) != col_map.end()) {
                Bj.push_back(col_map[col]);
                Bx.push_back(Ax[jj]);
                nnz++;
            }
        }
        Bp[i + 1] = nnz;
    }
}

template <typename I, typename T>
void csr_find_nonzero(const I n_row, const I Ap[], const I Aj[], const T Ax[],
                      std::vector<I>& row_indices, std::vector<I>& col_indices, std::vector<T>& values) {
    row_indices.clear();
    col_indices.clear();
    values.clear();

    for (I i = 0; i < n_row; i++) {
        for (I jj = Ap[i]; jj < Ap[i + 1]; jj++) {
            if (Ax[jj] != static_cast<T>(0)) {  // 仅存储非零元素
                row_indices.push_back(i);
                col_indices.push_back(Aj[jj]);
                values.push_back(Ax[jj]);
            }
        }
    }
}

template <typename I, typename T>
void csr_transpose(const I n_row, const I n_col, const I Ap[], const I Aj[], const T Ax[],
                   std::vector<I>& Bp, std::vector<I>& Bj, std::vector<T>& Bx) {
    // 1. 统计每列（即转置后每行）的非零元素个数
    Bp.assign(n_col + 1, 0);
    for (I i = 0; i < Ap[n_row]; i++) {
        Bp[Aj[i] + 1]++;  // 这里修正，直接统计列出现的次数
    }

    // 2. 计算累积和，得到新的行指针
    for (I i = 0; i < n_col; i++) {
        Bp[i + 1] += Bp[i];
    }

    // 3. 初始化存储列索引和数值
    Bj.resize(Ap[n_row]);
    Bx.resize(Ap[n_row]);

    // 4. 复制 Bp 用于追踪每个插入位置
    std::vector<I> next(Bp.begin(), Bp.end());

    // 5. 遍历原始 CSR 矩阵，构造转置矩阵
    for (I i = 0; i < n_row; i++) {
        for (I jj = Ap[i]; jj < Ap[i + 1]; jj++) {
            I col = Aj[jj];       // 取当前元素的列索引
            I dest = next[col]++; // 在新矩阵的对应行插入数据

            Bj[dest] = i;         // 变换后行变列
            Bx[dest] = Ax[jj];    // 保持数值不变
        }
    }
}


// 计算 CSR 矩阵的每列最小值
template <typename I, typename T>
void csr_column_min(const I n_row, const I n_col, const I Ap[], const I Aj[], const T Ax[],
                    std::vector<T>& min_values) {
    min_values.assign(n_col, std::numeric_limits<T>::max());

    for (I i = 0; i < n_row; i++) {
        for (I jj = Ap[i]; jj < Ap[i + 1]; jj++) {
            I col = Aj[jj];
            min_values[col] = std::min(min_values[col], Ax[jj]);
        }
    }

    // 处理没有元素的列
    for (I i = 0; i < n_col; i++) {
        if (min_values[i] == std::numeric_limits<T>::max()) {
            min_values[i] = 0.05;  // 或者设为 NaN，取决于需求
        }
    }
}

// 显式实例化
template void csr_column_min<int, double>(const int, const int, const int[], const int[], const double[],
                                          std::vector<double>&);


// 显式实例化
template void csr_transpose<int, double>(const int, const int, const int[], const int[], const double[],
                                         std::vector<int>&, std::vector<int>&, std::vector<double>&);


template void csr_find_nonzero<int, double>(const int, const int[], const int[], const double[],
    std::vector<int>&, std::vector<int>&, std::vector<double>&);

// 显式实例化
template void csr_extract_submatrix<int, double>(const int, const int[], const int[], const double[], 
                                                 const std::vector<int>&, const std::vector<int>&, 
                                                 std::vector<int>&, std::vector<int>&, std::vector<double>&);


// 显式实例化
template void csr_row_index<int, double>(const int, const int[], const int[], const int[], const double[], std::vector<int>&, std::vector<double>&);
template void csr_row_slice<int, double>(const int, const int, const int, const int[], const int[], const double[], std::vector<int>&, std::vector<double>&);
template void csr_column_index1<int>(const int, const int[], const int, const int, const int[], const int[], std::vector<int>&, std::vector<int>&);
template void csr_column_index2<int, double>(const int[], const int[], const int, const int[], const double[], std::vector<int>&, std::vector<double>&);
