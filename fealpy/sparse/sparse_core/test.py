import numpy as np
import csr_utils

Ap = np.array([0, 2, 4, 6], dtype=int)
Aj = np.array([0, 1, 1, 2, 0, 2], dtype=int)
Ax = np.array([2, 3, 3, 4, 4, 5], dtype=float)
rows = np.array([0, 2], dtype=int)
cols = np.array([1, 2], dtype=int)

Bp, Bj, Bx = csr_utils.csr_extract_submatrix(Ap, Aj, Ax, rows, cols)
print(Bp, Bj, Bx)

import numpy as np
import csr_utils

Ap = np.array([0, 2, 4, 7], dtype=np.int32)
Aj = np.array([0, 2, 1, 2, 0, 1, 2], dtype=np.int32)
Ax = np.array([1.0, 2.0, 0, 4.0, 5.0, 6.0, 0.0], dtype=np.float64)

row_indices, col_indices, values = csr_utils.csr_find_nonzero(Ap, Aj, Ax)

print("Row indices:", row_indices)
print("Col indices:", col_indices)
print("Values:", values)

import numpy as np
import csr_utils

# CSR 矩阵
Ap = np.array([0, 2, 4, 7], dtype=np.int32)   # 行指针
Aj = np.array([0, 2, 1, 2, 0, 1, 2], dtype=np.int32)  # 列索引
Ax = np.array([4.0, 5.0, -1.0, 3.0, 2.0, 6.0, 8.0], dtype=np.float64)  # 值
n_row, n_col = 3, 3  # 3x4 矩阵

# 计算转置
Bp, Bj, Bx = csr_utils.csr_transpose(Ap, Aj, Ax, n_row, n_col)
print("Bp:", Bp)
print("Bj:", Bj)
print("Bx:", Bx)

# 计算每列最小值
col_min = csr_utils.csr_column_min(Ap, Aj, Ax, n_row, n_col)
print("Column min values:", col_min)
