from fealpy.backend import backend_manager as bm
from fealpy.sparse.ops import spdiags
from fealpy.sparse import csr_matrix

def ruge_stuben_interpolation(isC,Am):
    N = Am.shape[0]
    allNode = bm.arange(N)
    fineNode = allNode[~isC]
    Nf = len(fineNode)
    Nc = N - Nf
    coarseNode = bm.arange(Nc)
    coarse2fine = bm.where(isC)[0]
    fine2coarse = bm.zeros(N, dtype=int)
    fine2coarse[isC] = coarseNode
    ip = coarse2fine
    jp = coarseNode
    sp_vals = bm.ones(Nc)

    Afc = Am[fineNode, coarse2fine]

    Dsum = 1 / bm.array(Afc.sum(axis=0)+0.1).flatten()
    k = Dsum.shape[0]
    indptr = bm.arange(k)
    indices = bm.arange(k)
    Dsum = csr_matrix((Dsum, (indices, indptr)), shape=(k, k))
    
    ti, tj, tw= (Dsum @ Afc).find()
    #tw = (Dsum @ Afc).data

    ip = bm.concatenate((ip, fineNode[ti]))
    jp = bm.concatenate((jp, tj))
    sp_vals = bm.concatenate((sp_vals, tw))
    Pro = csr_matrix((sp_vals, (ip, jp)), shape=(N, Nc))
    Res = Pro.T

    return Pro, Res

def standard_interpolation(A, isC):
    """
    @brief Generate prolongation and restriction matrices

    @param[in] A A symmetric positive definite matrix
    @param[in] isC A boolean array marking the coarse points

    @note The Prolongation matrix interpolates the solution from the coarse grid to the fine grid;
          The Restriction matrix restricts the residual from the fine grid to the coarse grid.
    """

    N = A.shape[0]

    # 1. Index mapping: The function first creates an index mapping from the coarse grid to the fine grid.
    #    It identifies all coarse and fine nodes and stores their indices.
    allNode = bm.arange(N)
    fineNode = allNode[~isC]
    NC = N - len(fineNode)
    coarseNode = bm.arange(NC)
    coarseNodeFineIdx = bm.nonzero(isC)[0]

    # 2. Construct prolongation and restriction operators
    Acf = A[coarseNodeFineIdx, fineNode]  # Extract the coarse-to-fine matrix block
    Dsum = bm.asarray(Acf.sum(axis=0)).reshape(-1)  # Sum of values corresponding to each fine node
    flag = (Dsum != 0)  # Boolean array marking fine nodes with nonzero sums
    NF = bm.sum(flag)  # Number of fine nodes
    Dsum = spdiags(1./Dsum[flag], diags=0, M=NF, N=NF)  # Form a sparse diagonal matrix
    flag = bm.nonzero(flag)[0]
    i, j, w = (Acf[:, flag] @ Dsum).find()  # Normalize each column by its sum
    # Note: 'j' represents fine node indices, 'i' represents coarse node indices
    # The prolongation matrix transfers information from the coarse grid to the fine grid
    I = bm.concatenate((coarseNodeFineIdx, fineNode[j]))
    J = bm.concatenate((coarseNode, i))
    val = bm.concatenate((bm.ones(NC), w))
    P = csr_matrix((val, (I, J)), shape=(N, NC))
    R = P.T
    return P, R



def rs_direct_interpolation(A, S, splitting):

    n_nodes = A.shape[0]
    
    Ap, Aj, Ax = A.indptr, A.indices, A.data
    Sp, Sj, Sx = S.indptr, S.indices, S.data


    Pp = bm.zeros(n_nodes + 1, dtype=int)
    nnz = 0
    
    for i in range(n_nodes):
        if splitting[i] == True:
            nnz += 1  # C 点插值矩阵 P 只连接自身
        else:
            for jj in range(Sp[i], Sp[i+1]):
                if splitting[Sj[jj]] == True and Sj[jj] != i:
                    nnz += 1  # F 点插值自邻近的 C 点
        
        Pp[i + 1] = nnz  # 记录 P 的非零元个数（CSR 结构）

    # **第二步：计算 P 的列索引 (Pj) 和数值 (Px)**
    Pj = bm.zeros(nnz, dtype=int)
    Px = bm.zeros(nnz, dtype=float)

    for i in range(n_nodes):
        if splitting[i] == True:
            Pj[Pp[i]] = i
            Px[Pp[i]] = 1.0  # C 点直接映射到自身
        else:
            # 计算 F 点插值的权重
            sum_strong_pos, sum_strong_neg = 0, 0
            for jj in range(Sp[i], Sp[i+1]):
                if splitting[Sj[jj]] == True and Sj[jj] != i:
                    if Sx[jj] < 0:
                        sum_strong_neg += Sx[jj]
                    else:
                        sum_strong_pos += Sx[jj]

            sum_all_pos, sum_all_neg, diag = 0, 0, 0
            for jj in range(Ap[i], Ap[i+1]):
                if Aj[jj] == i:
                    diag += Ax[jj]
                else:
                    if Ax[jj] < 0:
                        sum_all_neg += Ax[jj]
                    else:
                        sum_all_pos += Ax[jj]
            # 计算强连接的 C 点贡献
            # mask_c_neighbors = (splitting[Sj[Sp[i]:Sp[i+1]]] == True) & (Sj[Sp[i]:Sp[i+1]] != i)
            # strong_vals = Sx[Sp[i]:Sp[i+1]][mask_c_neighbors]

            # sum_strong_neg = strong_vals[strong_vals < 0].sum() if strong_vals.size > 0 else 0
            # sum_strong_pos = strong_vals[strong_vals > 0].sum() if strong_vals.size > 0 else 0

            # # 计算所有邻接点贡献
            # row_vals = Ax[Ap[i]:Ap[i+1]]
            # row_cols = Aj[Ap[i]:Ap[i+1]]

            # diag = row_vals[row_cols == i].sum()
            # sum_all_neg = row_vals[row_vals < 0].sum()
            # sum_all_pos = row_vals[row_vals > 0].sum()


            alpha = sum_all_neg / sum_strong_neg if sum_strong_neg != 0 else 0
            beta = sum_all_pos / sum_strong_pos if sum_strong_pos != 0 else 0

            if sum_strong_pos == 0:
                diag += sum_all_pos
                beta = 0

            neg_coeff = -alpha / diag if diag != 0 else 0
            pos_coeff = -beta / diag if diag != 0 else 0

            # 计算 P 的插值系数
            nnz_index = Pp[i]
            for jj in range(Sp[i], Sp[i+1]):
                if splitting[Sj[jj]] == True and Sj[jj] != i:
                    Pj[nnz_index] = Sj[jj]
                    Px[nnz_index] = neg_coeff * Sx[jj] if Sx[jj] < 0 else pos_coeff * Sx[jj]
                    nnz_index += 1

    coarse_map = bm.cumsum(splitting) - 1  # 计算 C 点编号映射
    Pj[:] = coarse_map[Pj]

    n_coarse = bm.sum(splitting)  # C 点个数
    P = csr_matrix((Px, Pj, Pp), shape=(n_nodes, n_coarse))
    r = P.T  
    return P,r

# import numpy as np

# def rs_direct_interpolation(A, S, splitting):
#     """
#     计算 Ruge-Stuben 直接插值矩阵 P（CSR 格式），尽量不使用显式 for 循环。

#     参数：
#       A        : scipy.sparse.csr_matrix  - 原始矩阵 (n x n)
#       S        : scipy.sparse.csr_matrix  - 强连接矩阵 (n x n)
#       splitting: array, shape (n,)         - C/F 划分数组（True 表示 C 点，False 表示 F 点）

#     返回：
#       P : scipy.sparse.csr_matrix - 插值矩阵 (n x n_coarse)（细到粗）
#       r : scipy.sparse.csr_matrix - R 矩阵，P.T
#     """
#     n_nodes = A.shape[0]
#     # CSR结构
#     Ap, Aj, Ax = A.indptr, A.indices, A.data
#     Sp, Sj, Sx = S.indptr, S.indices, S.data

#     # --- 第一步：构造 P 的行非零计数 ---
#     # 对于每一行 i：
#     #   若 splitting[i]==True (C 点)，计数为1；
#     #   否则 (F 点)，计数为：S 中 i 行满足 (neighbor is C and neighbor != i) 的个数。
#     # 首先，计算 S 中每行的“有效”邻居数。
#     row_inds_S = np.repeat(np.arange(n_nodes), np.diff(Sp))
#     valid_S = ((Sj != row_inds_S) & (splitting[Sj]))  # 对于每个非零，只有当对应的 neighbor 为 C 点且非对角时才有效
#     count_F = np.bincount(row_inds_S, weights=valid_S.astype(int), minlength=n_nodes)
#     # 对于 C 点，规定计数为 1。
#     row_counts = np.where(splitting, 1, count_F)
#     # 生成 P 的行指针数组
#     Pp = np.concatenate(([0], np.cumsum(row_counts)))
#     nnz = Pp[-1]

#     # --- 第二步：构造 P 的列索引和数值 ---
#     # 对于 C 点 i：直接设置列索引为 i，数值为 1。
#     C_nodes = np.nonzero(splitting)[0]
#     C_rows = C_nodes
#     C_cols = C_nodes  # 初步列索引（后面会重映射）
#     C_vals = np.ones_like(C_nodes, dtype=Ax.dtype)
    
#     # 对于 F 点，需要从 S 的数据中提取有效项，并计算插值权重。
#     # 首先，计算 A 中每行的对角元素和正、负系数
#     row_inds_A = np.repeat(np.arange(n_nodes), np.diff(Ap))
#     is_diag = (Aj == row_inds_A)
#     # 每行对角线（假设每行只有一个对角元素）
#     diag = np.bincount(row_inds_A, weights=Ax * is_diag.astype(Ax.dtype), minlength=n_nodes)
#     all_pos = np.where(Ax > 0, Ax, 0)
#     all_neg = np.where(Ax < 0, Ax, 0)
#     sum_all_pos = np.bincount(row_inds_A, weights=all_pos, minlength=n_nodes)
#     sum_all_neg = np.bincount(row_inds_A, weights=all_neg, minlength=n_nodes)
    
#     # 对 S：利用 row_inds_S 计算每行强连接中正、负部分的和
#     strong_pos = np.where((Sx > 0) & (Sj != row_inds_S) & (splitting[Sj]), Sx, 0)
#     strong_neg = np.where((Sx < 0) & (Sj != row_inds_S) & (splitting[Sj]), Sx, 0)
#     sum_strong_pos = np.bincount(row_inds_S, weights=strong_pos, minlength=n_nodes)
#     sum_strong_neg = np.bincount(row_inds_S, weights=strong_neg, minlength=n_nodes)
    
#     # 计算每行的系数 alpha, beta
#     alpha = np.where(sum_strong_neg != 0, sum_all_neg / sum_strong_neg, 0)
#     beta  = np.where(sum_strong_pos != 0, sum_all_pos / sum_strong_pos, 0)
#     # 对于没有强正连接的行，令 beta=0，同时调节 diag
#     diag_adj = np.where(sum_strong_pos == 0, diag + sum_all_pos, diag)
#     beta = np.where(sum_strong_pos == 0, 0, beta)
#     neg_coeff = np.where(diag_adj != 0, -alpha / diag_adj, 0)
#     pos_coeff = np.where(diag_adj != 0, -beta / diag_adj, 0)
    
#     # 对于 F 点，从 S 中提取有效的插值项
#     # 有效条件：行 i 为 F (splitting==False)，对应 S 非零的行 i： row_inds_S
#     # 并且对应的邻居 j 必须为 C（splitting[Sj]==True）且 j != i.
#     valid_mask = (~splitting[row_inds_S]) & (splitting[Sj]) & (Sj != row_inds_S)
#     F_rows = row_inds_S[valid_mask]   # F 点行号
#     F_orig_cols = Sj[valid_mask]        # 原始邻居（C 点）的编号
#     Svals_valid = Sx[valid_mask]         # 对应的 S 权重
    
#     # 对于每个 F 点 i, 插值权重由 S 权重乘以对应的系数（根据 Sx 正负分别选用 neg_coeff 或 pos_coeff）
#     F_vals = np.where(Svals_valid < 0, neg_coeff[F_rows] * Svals_valid, pos_coeff[F_rows] * Svals_valid)
    
#     # --- 合并 C 点和 F 点的数据 ---
#     # C 点：行 = C_rows, 列 = C_cols, data = C_vals
#     # F 点：行 = F_rows, 列 = F_orig_cols, data = F_vals
#     row_P = np.concatenate([C_rows, F_rows])
#     col_P = np.concatenate([C_cols, F_orig_cols])
#     data_P = np.concatenate([C_vals, F_vals])
    
#     # 将插值矩阵的条目按行排序（以便构造 CSR）
#     order = np.argsort(row_P, kind='mergesort')  # 使用稳定排序
#     row_P = row_P[order]
#     col_P = col_P[order]
#     data_P = data_P[order]
    
#     # --- 第三步：重新映射 P 的列索引，使之与粗网格编号对应 ---
#     # 构造 coarse_map：对于每个节点 i，如果 splitting[i]==True，则其粗网格编号为：cumsum(splitting.astype(int)) - 1.
#     coarse_map = np.cumsum(splitting.astype(int)) - 1
#     col_P = coarse_map[col_P]
    
#     # --- 构造稀疏矩阵 ---
#     n_coarse = int(np.sum(splitting))
#     # 采用 COO 格式，然后转换为 CSR 格式
#     P = csr_matrix((data_P, (row_P, col_P)), shape=(n_nodes, n_coarse))
#     r = P.T
#     return P, r


from .amg_core import amg_core
def rs_direct_interpolation_1(A, S, splitting):
    n_nodes = A.shape[0]
    
    # CSR 结构
    Ap, Aj, Ax = A.indptr, A.indices, A.data
    Sp, Sj, Sx = S.indptr, S.indices, S.data

    Pp = bm.zeros(n_nodes + 1, dtype=bm.int32)
    amg_core.rs_direct_interpolation_pass1(n_nodes, Sp, Sj, splitting, Pp)

    nnz = Pp[-1]  # 非零元素个数
    Pj = bm.empty(nnz, dtype=bm.int32)  # 预分配索引数组
    Px = bm.empty(nnz, dtype=bm.float64)  # 预分配数据数组

    amg_core.rs_direct_interpolation_pass2(n_nodes, A.indptr, A.indices, A.data,
                                       S.indptr, S.indices, S.data,
                                       splitting, Pp, Pj, Px)
    # amg_core.rs_classical_interpolation_pass2(n_nodes, A.indptr, A.indices, A.data,
    #                                    S.indptr, S.indices, S.data,
    #                                    splitting, Pp, Pj, Px,True)
    nc = bm.sum(splitting)
    p = csr_matrix((Px, Pj, Pp), shape=[n_nodes, nc])
    r = p.T
    return p, r
