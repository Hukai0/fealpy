
from typing import Optional, Protocol
import math

from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..sparse.csr_tensor import CSRTensor
from .. import logger

class SupportsMatmul(Protocol):
    def __matmul__(self, other: TensorLike) -> TensorLike: ...
    
class IncompleteCholesky:
    r"""
    Incomplete Cholesky factorization for a symmetric positive definite matrix.

    Given a symmetric positive definite matrix A in CSR format, this class computes a
    lower triangular factor L (stored in CSR format) such that ideally A ≈ L Lᵀ.
    A dropping strategy based on a threshold is used to control fill-in.

    Attributes:
        matA (CSRTensor): The input symmetric positive definite matrix.
        threshold (float): Dropping threshold factor.
        L_data (list[float]): Nonzero values of the computed lower triangular factor L.
        L_indices (list[int]): Column indices corresponding to L_data.
        L_indptr (list[int]): Row pointer array of L in CSR format.
        is_ready (bool): Flag indicating whether the factorization has been computed.
        Mnnz (int): An estimate of the fill-in (nonzero count) in the preconditioner.
    """

    def __init__(self, matA: SupportsMatmul, threshold: float) -> None:
        r"""
        Initialize the IncompleteCholesky preconditioner.

        Parameters:
            matA (CSRTensor): A symmetric positive definite matrix in CSR format.
            threshold (float): Dropping threshold factor for small values.
        """
        self.matA = matA
        self.threshold = threshold
        self.ArowNum, self.AcolNum = self.matA.shape
        if self.ArowNum != self.AcolNum:
            raise ValueError("The number of rows and columns must be the same!")
        # 假设 CSR 数据结构中 indptr 的最后一个元素表示非零元个数
        self.Annz = self.matA.indptr[-1]
        # 存放 L 的数据，采用列表方式构造（下三角部分）
        self.L_data: list[float] = []
        self.L_indices: list[int] = []
        self.L_indptr: list[int] = [0] * (self.ArowNum + 1)
        self.is_ready: bool = False
        self.Mnnz: int = 0

    def setup(self) -> None:
        r"""
        Compute the incomplete Cholesky factorization of A.

        This function computes the factor L such that A ≈ L Lᵀ, applying a dropping
        strategy based on the threshold. The factor L is stored in CSR format.
        """
        if self.is_ready:
            return

        n = self.ArowNum
        # 临时变量，均使用 Python 内置列表方式存储
        vUsedCol = [False] * n
        vIdxCol = [None] * n         # type: list[Optional[int]]
        vValCol = [0.0] * n
        vNonZeroRow = [None] * n       # 用于链接非零元素，初始为 None
        vNextIdxRow = [None] * n

        A_indptr = self.matA.indptr
        A_indices = self.matA.indices
        A_data = self.matA.data

        numNz = 0  # 当前 L 中非零元个数
        self.L_indptr[0] = 0

        for jj in range(n):
            n_indices = 0
            colThr = 0.0
            # 遍历 A 的第 jj 行（只考虑下三角部分，即 ii >= jj）
            for idx in range(A_indptr[jj], A_indptr[jj + 1]):
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
                L_jk = self.L_data[idx_start]
                for idx in range(idx_start, self.L_indptr[kk + 1]):
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
                if idx_start < self.L_indptr[kk + 1]:
                    vNextIdxRow[kk] = idx_start
                    ii = self.L_indices[idx_start]
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

            # 将 vIdxCol 中的列索引排序
            sorted_idx = sorted(vIdxCol[:n_indices])  # type: ignore
            first = True
            for ii in sorted_idx:
                if ii != jj:
                    L_ij = vValCol[ii]
                    if abs(L_ij) >= colThr:
                        if first:
                            first = False
                            vNonZeroRow[jj] = vNonZeroRow[ii]
                            vNonZeroRow[ii] = jj
                        L_ij = L_ij / diagElement
                        self.L_data.append(L_ij)
                        self.L_indices.append(ii)
                        numNz += 1
                vUsedCol[ii] = False
            self.L_indptr[jj + 1] = numNz

        self.is_ready = True
        self.Mnnz = numNz * 2 + n

    def MSolveLowerUsePtr(self, vec: TensorLike) -> None:
        r"""
        Solve the lower triangular system L y = vec using forward substitution.

        The solution overwrites the input vector 'vec'.

        Parameters:
            vec (TensorLike): Right-hand side vector; on output, contains the solution y.
        """
        if not self.is_ready:
            raise ValueError("The IC preconditioner is not ready!")
        n = self.ArowNum
        # 前向代回
        for i in range(n):
            k = self.L_indptr[i]
            Lii = self.L_data[k]
            tmpVal = vec[i] / Lii
            vec[i] = tmpVal  # y[i]
            for k in range(self.L_indptr[i] + 1, self.L_indptr[i + 1]):
                j = self.L_indices[k]
                Lij = self.L_data[k]
                vec[j] -= Lij * tmpVal

    def MSolveUpperUsePtr(self, vec: TensorLike) -> None:
        r"""
        Solve the upper triangular system Lᵀ x = y using backward substitution.

        The solution overwrites the input vector 'vec'.

        Parameters:
            vec (TensorLike): Right-hand side vector (typically y from forward substitution);
                              on output, contains the solution x.
        """
        if not self.is_ready:
            raise ValueError("The IC preconditioner is not ready!")
        n = self.ArowNum
        # 后向代回
        for i in range(n - 1, -1, -1):
            tmpVal = vec[i]
            for k in range(self.L_indptr[i] + 1, self.L_indptr[i + 1]):
                j = self.L_indices[k]
                Lij = self.L_data[k]
                tmpVal -= Lij * vec[j]
            Lii = self.L_data[self.L_indptr[i]]
            vec[i] = tmpVal / Lii

