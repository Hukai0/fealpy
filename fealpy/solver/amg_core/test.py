import math
import numpy as np
from scipy.sparse import csr_matrix

class IncompleteCholesky:
    def __init__(self, matA: csr_matrix, threshold: float):
        """
        matA: 输入矩阵，要求为对称正定矩阵，使用 csr_matrix 格式
        threshold: 舍弃小值的阈值因子
        """
        self.matA = matA  # 假定为 scipy.sparse.csr_matrix
        self.threshold = threshold
        self.ArowNum, self.AcolNum = matA.shape
        if self.ArowNum != self.AcolNum:
            raise ValueError("The number of rows and columns must be the same!")
        # 非零元个数
        self.Annz = matA.indptr[-1]
        # 预先构造存放不完全 Cholesky 分解结果的矩阵（以 CSR 格式存储下三角矩阵 L）
        # 我们采用列表累积方式构造 CSR 数据：L_data, L_indices, L_indptr
        self.L_data = []      # 存储所有非零值
        self.L_indices = []   # 存储对应列索引
        # 长度为 n+1 的行偏移指针数组
        self.L_indptr = [0] * (self.ArowNum + 1)
        self.is_ready = False
        self.Mnnz = 0

    def setup(self):
        """
        执行不完全 Cholesky 分解
        """
        if self.is_ready:
            return

        n = self.ArowNum
        # 临时数组
        vUsedCol = np.zeros(n, dtype=bool)
        # 使用列表存储待处理的列索引，预分配大小 n（后续只使用前 n_indices 部分）
        vIdxCol = [None] * n
        vValCol = np.zeros(n, dtype=np.float64)
        # 用于合并排序的缓冲区在 Python 中可直接使用 sorted() 实现
        # 下面两个数组用于记录非零元素的链接关系，初值使用 None 作为哨兵（对应 C++ 中 UINT32_MAX）
        vNonZeroRow = [None] * n
        vNextIdxRow = [None] * n

        # 取出 A 的 CSR 数据
        A_indptr = self.matA.indptr
        A_indices = self.matA.indices
        A_data = self.matA.data

        numNz = 0  # 当前 L 中非零元的个数
        self.L_indptr[0] = 0

        # 遍历 A 的每一行（对应对称矩阵的第 j 列/行）
        for jj in range(n):
            n_indices = 0
            colThr = 0.0
            # 遍历 A 的第 jj 行（只考虑下三角部分，即 ii >= jj）
            for idx in range(A_indptr[jj], A_indptr[jj+1]):
                ii = A_indices[idx]
                A_ij = A_data[idx]
                if ii >= jj:
                    colThr += abs(A_ij)
                    if not vUsedCol[ii]:
                        vValCol[ii] = 0.0
                        vUsedCol[ii] = True
                        vIdxCol[n_indices] = ii
                        n_indices += 1
                    vValCol[ii] += A_ij
            colThr *= self.threshold

            # 利用已计算的 L 因子更新当前列的数据
            kk = vNonZeroRow[jj]
            while kk is not None:
                idx_start = vNextIdxRow[kk]
                # 取出 L(j, k) 对应的值
                L_jk = self.L_data[idx_start]
                # 遍历第 kk 行中 idx_start 后面的所有非零元素
                for idx in range(idx_start, self.L_indptr[kk+1]):
                    ii = self.L_indices[idx]
                    L_ik = self.L_data[idx]
                    if not vUsedCol[ii]:
                        vUsedCol[ii] = True
                        vValCol[ii] = 0.0
                        vIdxCol[n_indices] = ii
                        n_indices += 1
                    vValCol[ii] -= L_ik * L_jk
                idx_start += 1
                k_next = vNonZeroRow[kk]
                if idx_start < self.L_indptr[kk+1]:
                    vNextIdxRow[kk] = idx_start
                    ii = self.L_indices[idx_start]
                    # 更新链表关系
                    vNonZeroRow[kk] = vNonZeroRow[ii]
                    vNonZeroRow[ii] = kk
                kk = k_next

            diagElement = vValCol[jj]
            if diagElement <= 0.0:
                raise ValueError("The original matrix is not positive definite!")
            diagElement = math.sqrt(diagElement)
            vValCol[jj] = diagElement

            # 将对角元素写入 L
            self.L_data.append(diagElement)
            self.L_indices.append(jj)
            numNz += 1
            vNextIdxRow[jj] = numNz

            # 将 vIdxCol 中存储的列索引排序
            sorted_idx = sorted(vIdxCol[:n_indices])
            first = True
            for ii in sorted_idx:
                if ii != jj:
                    L_ij = vValCol[ii]
                    # 舍弃小值
                    if abs(L_ij) >= colThr:
                        if first:
                            first = False
                            vNonZeroRow[jj] = vNonZeroRow[ii]
                            vNonZeroRow[ii] = jj
                        L_ij = L_ij / diagElement
                        self.L_data.append(L_ij)
                        self.L_indices.append(ii)
                        numNz += 1
                # 清除标记
                vUsedCol[ii] = False
            self.L_indptr[jj+1] = numNz

        self.is_ready = True
        self.Mnnz = numNz * 2 + n

    def MSolveLowerUsePtr(self, vec: np.ndarray):
        """
        使用不完全 Cholesky 分解得到的 L（存储在 self.L_* 中）解下三角方程 L y = vec，
        其中 vec 为右端项，解结果将直接覆盖在 vec 中。
        """
        if not self.is_ready:
            raise ValueError("The IC precondition is not ready!")
        n = self.ArowNum
        # 前向代回
        for i in range(n):
            k = self.L_indptr[i]
            Lii = self.L_data[k]
            tmpVal = vec[i] / Lii
            vec[i] = tmpVal  # 此时 vec[i] 为 y[i]
            for k in range(self.L_indptr[i] + 1, self.L_indptr[i+1]):
                j = self.L_indices[k]
                Lij = self.L_data[k]
                vec[j] -= Lij * tmpVal

    def MSolveUpperUsePtr(self, vec: np.ndarray):
        """
        使用不完全 Cholesky 分解得到的 L 进行上三角回代（实际上求解 Lᵀ x = y）；
        其中 vec 为右端项，解结果将直接覆盖在 vec 中。
        """
        if not self.is_ready:
            raise ValueError("The IC precondition is not ready!")
        n = self.ArowNum
        # 后向代回
        for i in range(n - 1, -1, -1):
            tmpVal = vec[i]
            for k in range(self.L_indptr[i] + 1, self.L_indptr[i+1]):
                j = self.L_indices[k]
                Lij = self.L_data[k]
                tmpVal -= Lij * vec[j]
            Lii = self.L_data[self.L_indptr[i]]
            vec[i] = tmpVal / Lii

# ===========================
# 示例用法
# ===========================
if __name__ == '__main__':
    # 构造一个对称正定的稀疏矩阵
    from scipy.sparse import diags

    n = 5
    # 例如对角线全为 4，次对角线全为 -1（典型的五点差分离散）
    diagonals = [4 * np.ones(n), -1 * np.ones(n - 1), -1 * np.ones(n - 1)]
    A = diags(diagonals, [0, -1, 1], format='csr')
    
    # 初始化不完全 Cholesky 分解预条件器，阈值设定为 0.01
    ic = IncompleteCholesky(A, threshold=0.01)
    ic.setup()
    
    # 构造一个右端项向量，进行下三角解和上三角解
    b = np.random.rand(n)
    # 为测试预条件器，我们先进行下三角回代，然后上三角回代
    vec_lower = b.copy()
    ic.MSolveLowerUsePtr(vec_lower)
    vec_upper = vec_lower.copy()
    ic.MSolveUpperUsePtr(vec_upper)
    
    print("Original b:", b)
    print("After lower and upper solve:", vec_upper)
    print("Residual:", A.dot(vec_upper) - b)
