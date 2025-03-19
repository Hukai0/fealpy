import numpy as np
import csr_utils  # 这应该是你的扩展模块

# 测试 csr_row_index
def test_csr_row_index():
    rows = np.array([0, 1, 2], dtype=int)
    Ap = np.array([0, 2, 4, 6], dtype=int)
    Aj = np.array([0, 1, 1, 2, 2, 3], dtype=int)
    Ax = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)

    Bj, Bx = csr_utils.csr_row_index(rows, Ap, Aj, Ax)
    print("csr_row_index")
    print("Bj:", Bj)
    print("Bx:", Bx)

# 测试 csr_row_slice
def test_csr_row_slice():
    Ap = np.array([0, 2, 4, 6], dtype=int)
    Aj = np.array([0, 1, 1, 2, 2, 3], dtype=int)
    Ax = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)

    start, stop, step = 0, 2, 1
    Bj, Bx = csr_utils.csr_row_slice(start, stop, step, Ap, Aj, Ax)
    print("csr_row_slice")
    print("Bj:", Bj)
    print("Bx:", Bx)

# 测试 csr_column_index1
def test_csr_column_index1():
    col_idxs = np.array([0, 1, 2], dtype=int)
    Ap = np.array([0, 2, 4, 6], dtype=int)
    Aj = np.array([0, 1, 1, 2, 2, 3], dtype=int)
    n_row = 3
    n_col = 4

    col_offsets, Bp = csr_utils.csr_column_index1(col_idxs, n_row, n_col, Ap, Aj)
    print("csr_column_index1")
    print("col_offsets:", col_offsets)
    print("Bp:", Bp)

# 测试 csr_column_index2
def test_csr_column_index2():
    col_order = np.array([0, 1, 2], dtype=int)
    col_offsets = np.array([0, 2, 4], dtype=int)
    nnz = 6
    Aj = np.array([0, 1, 1, 2, 2, 3], dtype=int)
    Ax = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)

    Bj, Bx = csr_utils.csr_column_index2(col_order, col_offsets, nnz, Aj, Ax)
    print("csr_column_index2")
    print("Bj:", Bj)
    print("Bx:", Bx)

# 运行测试
if __name__ == "__main__":
    test_csr_row_index()
    test_csr_row_slice()
    #test_csr_column_index1()
    test_csr_column_index2()
