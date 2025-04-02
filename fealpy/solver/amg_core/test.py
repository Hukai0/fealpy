import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def tridiagonal_preconditioner(A):
    """ 三对角预处理子 P1 = A - tril(A,-2) - triu(A,2) """
    P1 = A.copy()
    P1 = sp.tril(P1, k=1) + sp.triu(P1, k=-1)  # 保留三对角部分
    return P1

def lower_triangular_preconditioner(A):
    """ 下三角预处理子 P2 = tril(A,0) """
    return sp.tril(A, format='csr')  # 保留下三角部分（包括主对角线）

def ilu_preconditioner(A, droptol=0.1):
    """ 不完全 LU 分解预处理子 P3 = L * U """
    ilu = spla.spilu(A, drop_tol=droptol)
    M = spla.LinearOperator(A.shape, ilu.solve)  # 生成线性算子
    return M

# 生成二维泊松矩阵
def build_2d_poisson(l, q=10):
    h = 1.0 / (l + 1)
    main_diag = -2 * np.ones(l)
    off_diag = np.ones(l - 1)
    T = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csr')
    I = sp.eye(l, format='csr')
    
    A_lap = (1.0 / h**2) * (sp.kron(I, T) + sp.kron(T, I))
    A_pot = q * sp.eye(l * l, format='csr')
    A = A_lap + A_pot
    return A

# 迭代求解并记录迭代次数
def solve_with_gmres(A, b, M=None):
    x, info = spla.minres(A, b, M=M,rtol=1e-10)
    return info  # 迭代次数

if __name__ == '__main__':
    l = 100  # 网格点数
    q = 10  # 势项系数

    # 生成矩阵
    A = build_2d_poisson(l, q)
    print(A.data == A.T.data)  # 检查矩阵是否对称
    b = np.ones(A.shape[0])  # 右端项

    # 不同预处理子
    P1 = tridiagonal_preconditioner(A)
    P2 = lower_triangular_preconditioner(A)
    P3 = ilu_preconditioner(A)

    # 直接求解 GMRES
    iter_no_precond = solve_with_gmres(A, b)
    print(f"无预处理 GMRES 迭代次数: {iter_no_precond}")

    # 预处理 GMRES (三对角)
    iter_tri = solve_with_gmres(A, b, M=spla.LinearOperator(A.shape, lambda x: spla.spsolve(P1, x)))
    print(f"三对角预处理 GMRES 迭代次数: {iter_tri}")

    # 预处理 GMRES (下三角)
    # iter_lower = solve_with_gmres(A, b, M=spla.LinearOperator(A.shape, lambda x: spla.spsolve(P2, x)))
    # print(f"下三角预处理 GMRES 迭代次数: {iter_lower}")

    # 预处理 GMRES (ILU)
    iter_ilu = solve_with_gmres(A, b, M=P3)
    print(f"ILU 预处理 GMRES 迭代次数: {iter_ilu}")
