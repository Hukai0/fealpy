
from fealpy.backend import backend_manager as bm
from fealpy.sparse.ops import spdiags
from fealpy.sparse import csr_matrix
import time

# C/F splitting 的标记
U_NODE = -1  # 未标记
C_NODE = 1   # C-节点
F_NODE = 0   # F-节点
PRE_F_NODE = 2  # 临时 F-节点

def rs_cf_splitting_optimized(A):
    n_nodes = A.shape[0]
    U = bm.arange(n_nodes)
    import math
    k = int(math.log2(n_nodes))

    Sp, Sj = A.indptr, A.indices
    Tp, Tj = A.T.indptr, A.T.indices

    # 初始化lambda_vals
    lambda_vals = Tp[1:] - Tp[:-1]

    splitting = bm.full(n_nodes, U_NODE, dtype=int)
    
    mask = (lambda_vals == 0) | ((lambda_vals == 1) & bm.any(Tj[Tp[:-1]] == bm.arange(n_nodes)))

    splitting[mask] = F_NODE
    unodes = bm.sum(splitting == U_NODE)
 
    while (unodes > k):

        # indices = bm.nonzero(splitting == U_NODE)[0]  # 获取全局索引
        # i = indices[bm.argmax(lambda_vals[splitting == U_NODE])]  # 取出最大值对应的全局索引
        i = bm.argmax(lambda_vals)
        splitting[i] = C_NODE
        lambda_vals[i] = 0
        unodes -= 1

        # 处理强影响邻居
        j_list = Tj[Tp[i]:Tp[i+1]]
        j_mask = (splitting[j_list] == U_NODE)
        selected_j = j_list[j_mask]
        splitting[selected_j] = F_NODE
        lambda_vals[selected_j] = 0
        unodes -= len(selected_j)
        # 批量更新k的lambda值
        starts = Sp[selected_j]
        ends = Sp[selected_j + 1]
        #k_indices = bm.concatenate([Sj[s:e] for s, e in zip(starts, ends)])

        arrays = [bm.arange(s, e) for s, e in zip(starts, ends) if e > s]
        k_indices = Sj[bm.concatenate(arrays)] if arrays else bm.empty(0, dtype=int)
      
        
        k_mask = (splitting[k_indices] == U_NODE)

        k_to_update = k_indices[k_mask]
        unique_k = bm.unique(k_to_update)
        lambda_vals[unique_k] += 1

        # 批量处理C点的强邻居
        i_neighbors = Sj[Sp[i]:Sp[i+1]]
        j_mask = (splitting[i_neighbors] == U_NODE)
        j_to_decrease = i_neighbors[j_mask]
        lambda_vals[j_to_decrease] -= 1

    splitting[splitting == U_NODE] = F_NODE
    return splitting

def rs_cf_splitting(A):

    n_nodes = A.shape[0]
    

    Sp, Sj = A.indptr, A.indices
    Tp, Tj = A.T.indptr, A.T.indices  # 计算 S^T 的 CSR 结构
    
    
    lambda_vals = Tp[1:] - Tp[:-1]
    interval_ptr = bm.zeros(n_nodes + 1, dtype=int)
    # 有bug，默认连接点不超过n个
    interval_count = bm.zeros(n_nodes, dtype=int)

    interval_count[:max(lambda_vals) + 1] = bm.bincount(lambda_vals)
    interval_ptr[1:] = bm.cumsum(interval_count)

    sorted_indices = bm.argsort(lambda_vals)
    node_to_index = bm.arange(len(lambda_vals))
    node_to_index[sorted_indices] = bm.arange(len(lambda_vals))
    index_to_node = bm.argsort(node_to_index)

    splitting = bm.full(n_nodes, U_NODE, dtype=int)
    #mask = (lambda_vals == 0) | ((lambda_vals == 1) & bm.any(Tj[Tp[:-1]] == bm.arange(n_nodes)))
    #splitting[mask] = F_NODE
    for i in range(n_nodes):
        if lambda_vals[i] == 0 or (lambda_vals[i] == 1 and Tj[Tp[i]] == i):
            splitting[i] = F_NODE

    # 逐步划分 C/F 点
    for top_index in range(n_nodes - 1, -1, -1):
        i = index_to_node[top_index]
        lambda_i = lambda_vals[i]

        interval_count[lambda_i] -= 1

        if lambda_i <= 0:
            break  

        if splitting[i] == U_NODE:
            splitting[i] = C_NODE

            # 处理所有 PRE_F_NODE
            for jj in range(Tp[i], Tp[i+1]):
                j = Tj[jj]
                if splitting[j] == U_NODE:
                    splitting[j] = F_NODE

                    # 更新邻居 k 的 lambda 值
                    for kk in range(Sp[j], Sp[j+1]):
                        k = Sj[kk]
                        if splitting[k] == U_NODE:
                            if lambda_vals[k] >= n_nodes - 1:
                                continue
                            old_pos = node_to_index[k]
                            new_pos = interval_ptr[lambda_vals[k]] + interval_count[lambda_vals[k]] - 1

                            # 交换位置
                            node_to_index[index_to_node[old_pos]] = new_pos
                            node_to_index[index_to_node[new_pos]] = old_pos
                            index_to_node[old_pos], index_to_node[new_pos] = index_to_node[new_pos], index_to_node[old_pos]

                            # 更新间隔计数
                            interval_count[lambda_vals[k]] -= 1
                            interval_count[lambda_vals[k] + 1] += 1
                            interval_ptr[lambda_vals[k] + 1] = new_pos

                            # 增加 lambda 值
                            lambda_vals[k] += 1

            # 处理 S_i，降低邻居 j 的 lambda
            for jj in range(Sp[i], Sp[i+1]):
                j = Sj[jj]
                if splitting[j] == U_NODE:
                    if lambda_vals[j] == 0:
                        continue
                    old_pos = node_to_index[j]
                    new_pos = interval_ptr[lambda_vals[j]]

                    # 交换位置
                    node_to_index[index_to_node[old_pos]] = new_pos
                    node_to_index[index_to_node[new_pos]] = old_pos
                    index_to_node[old_pos], index_to_node[new_pos] = index_to_node[new_pos], index_to_node[old_pos]

                    # 更新间隔计数
                    interval_count[lambda_vals[j]] -= 1
                    interval_count[lambda_vals[j] - 1] += 1
                    interval_ptr[lambda_vals[j]] += 1
                    interval_ptr[lambda_vals[j] - 1] = interval_ptr[lambda_vals[j]] - interval_count[lambda_vals[j] - 1]

                    # 减小 lambda 值
                    lambda_vals[j] -= 1

    # 将所有未标记的节点设为 F_NODE
    splitting[splitting == U_NODE] = F_NODE

    return splitting
# import heapq
# def rs_cf_splitting(A):
#     """
#     RS 粗化算法优化实现：
#       - 使用 NumPy 布尔数组存储节点状态：
#             isC: C 点 (粗点)
#             isF: F 点 (细点)
#             isU: 未确定点 (U 集)
#       - 使用 CSR 格式的矩阵 A 直接访问邻接信息；
#       - 用 heapq 维护影响值 v（λ 值），存入 (-λ, i) 实现最大堆，加速最大值选择；
#       - 利用向量化与布尔索引更新邻居影响值。
      
#     参数：
#       A : scipy.sparse.csr_matrix
#           强连接矩阵 S (n x n)，其 CSR 数据：indptr, indices
      
#     返回：
#       splitting: array, shape (n,)
#           C/F 划分结果，其中 C_NODE = 1, F_NODE = -1.
#     """
#     n_nodes = A.shape[0]
    
#     # 获取 A 的 CSR 结构
#     Sp, Sj = A.indptr, A.indices
#     # 获取 A^T 的 CSR 结构（S^T），用以计算入度，λ 值
#     Tp, Tj = A.T.indptr, A.T.indices
    
#     # 计算每个节点的影响值 λ = |S^T_i|（即入度）
#     lambda_vals = Tp[1:] - Tp[:-1]  # shape (n_nodes,)
    
#     # 定义状态：U_NODE = 0, C_NODE = 1, F_NODE = -1
#     U_NODE, C_NODE, F_NODE = 0, 1, -1
    
#     # 使用布尔数组表示各状态
#     isC = bm.zeros(n_nodes, dtype=bool)  # 初始全为 False
#     isF = bm.zeros(n_nodes, dtype=bool)
#     # U 集：未确定节点，初始时全部为 True
#     isU = bm.ones(n_nodes, dtype=bool)
    
#     # 预处理：对于 λ==0 或 (λ==1 且该节点在 S^T 中只有自身) 的节点，直接设为 F 点
#     for i in range(n_nodes):
#         if lambda_vals[i] == 0 or (lambda_vals[i] == 1 and Tj[Tp[i]] == i):
#             isF[i] = True
#             isU[i] = False

#     # 构造最大堆：对于 U 中的每个节点 i，存入 (-λ, i)
#     heap = [(-lambda_vals[i], i) for i in range(n_nodes) if isU[i]]
#     heapq.heapify(heap)
    
#     # 主循环：不断从 U 中选取影响值最大的节点
#     while heap:
#         neg_val, i = heapq.heappop(heap)
#         # 如果该节点已经不在 U 中，跳过
#         if not isU[i]:
#             continue
#         # 选择 i 为 C 点
#         isC[i] = True
#         isU[i] = False
        
#         # 处理 S^T 中 i 的邻居：
#         # 取出 i 在 A^T 中的邻接点 j：即 Tj[Tp[i]:Tp[i+1]]
#         j_candidates = Tj[Tp[i]:Tp[i+1]]
#         # 只考虑仍在 U 中的 j
#         valid_j = j_candidates[isU[j_candidates]]
#         if valid_j.size > 0:
#             # 标记这些 j 为 F 点
#             isF[valid_j] = True
#             isU[valid_j] = False
#             # 对于每个这样的 j，更新其邻居的影响值（增加 1）\n
#             # 利用 CSR 格式，从 A 的行 j 中取出邻居（即 Sj[Sp[j]:Sp[j+1]]）\n
#             # 这里采用列表解析合并所有 j 的邻居，然后用布尔索引更新\n
#             arrays = [Sj[Sp[j]:Sp[j+1]] for j in valid_j if Sp[j] < Sp[j+1]]
#             if arrays:
#                 neighbor_indices = bm.concatenate(arrays)
#                 # 仅对仍在 U 中的邻居更新影响值
#                 update_mask = isU[neighbor_indices]
#                 indices_to_update = neighbor_indices[update_mask]
#                 if indices_to_update.size > 0:
#                     lambda_vals[indices_to_update] += 1
#                     # 将更新后的节点重新入堆
#                     for k in indices_to_update:
#                         heapq.heappush(heap, (-lambda_vals[k], k))
        
#         # 处理 A 中 i 的邻居（即 S 的行 i），对仍在 U 中的邻居 j 降低影响值 1
#         j_candidates2 = Sj[Sp[i]:Sp[i+1]]
#         valid_j2 = j_candidates2[isU[j_candidates2]]
#         if valid_j2.size > 0:
#             lambda_vals[valid_j2] -= 1
#             for j in valid_j2:
#                 heapq.heappush(heap, (-lambda_vals[j], j))
    
#     # 最后，将剩余 U 中的节点全部标记为 F 点
#     isF[isU] = True
#     isU[:] = False
    
#     # 构造最终 splitting 数组：C_NODE 对应 1，F_NODE 对应 -1
#     splitting = bm.where(isC, C_NODE, F_NODE)
#     return splitting

from .amg_core import amg_core
def rs_cf_splitting_1(A):
    splitting = amg_core.rs_cf_splitting(A.indptr, A.indices, A.T.indptr, A.T.indices)
    return splitting
    

# def classical_strength_of_connection( A, theta=0.25):
#     """
#     计算经典强连接矩阵 S（CSR 格式）
    
#     参数：
#     n_row : int                      - 矩阵 A 的行数
#     theta : float                    - 强连接阈值参数
#     A     : scipy.sparse.csr_matrix  - 原始矩阵 A (n x n)

#     返回：
#     S     : scipy.sparse.csr_matrix  - 强连接矩阵 S (n x n)
#     """
#     # 获取 A 的 CSR 结构
#     n_row = A.shape[0]
#     Ap, Aj, Ax = A.indptr, A.indices, A.data

#     # 预分配 CSR 存储结构
#     Sp = bm.zeros(n_row + 1, dtype=int)
#     Sj = []
#     Sx = []
    
#     nnz = 0
#     Sp[0] = 0

#     for i in range(n_row):
#         row_start, row_end = Ap[i], Ap[i+1]

#         # 计算非对角元素的最大范数
#         max_offdiagonal = 0  # min 用于浮点数安全
#         for jj in range(row_start, row_end):
#             if Aj[jj] != i:
#                 max_offdiagonal = max(max_offdiagonal, abs(Ax[jj]))

#         # 计算阈值
#         threshold = theta * max_offdiagonal

#         for jj in range(row_start, row_end):
#             norm_jj = abs(Ax[jj])

#             # 仅当满足阈值条件时，添加非对角元素
#             if norm_jj >= threshold and Aj[jj] != i:
#                 Sj.append(Aj[jj])
#                 Sx.append(Ax[jj])
#                 nnz += 1

#             # 始终添加对角元素
#             if Aj[jj] == i:
#                 Sj.append(Aj[jj])
#                 Sx.append(Ax[jj])
#                 nnz += 1

#         Sp[i+1] = nnz  # 记录行指针
    
#     # 转换为 numpy 数组
#     Sj = bm.array(Sj, dtype=int)
#     Sx = bm.array(Sx, dtype=A.dtype)

#     # 构造 CSR 格式的强连接矩阵 S
#     S = csr_matrix((Sx, Sj, Sp), shape=(n_row, n_row))
#     # from line_profiler import LineProfiler
#     # lp = LineProfiler()
#     # lp.add_function(rs_cf_splitting)
#     # lp.enable()
#     splitting = rs_cf_splitting(S)
#     # lp.disable()
#     # lp.print_stats()

#     return splitting,S
def classical_strength_of_connection(A, theta=0.25):

    n_row = A.shape[0]
    Ap, Aj, Ax = A.indptr, A.indices, A.data

    # 为每个非零元素构造对应的行索引数组
    # 每一行重复行号，其重复次数等于该行非零数目 np.diff(Ap)
    import numpy as np
    row_inds = bm.repeat(bm.arange(n_row), np.diff(Ap))
    absAx = bm.abs(Ax)
    
    # 对于非对角元素 (Aj != row_inds) 计算绝对值；对角元素置为0
    is_offdiag = (Aj != row_inds)
    offdiag_abs = absAx.copy()
    offdiag_abs[~is_offdiag] = 0

    # 利用 np.maximum.reduceat 按行计算非对角元素的最大值
    # 注意：每行至少包含一个对角元素，所以该行不会为空
    max_off = bm.maximum.reduceat(offdiag_abs, Ap[:-1])

    # 计算每行的阈值：theta * (最大非对角绝对值)
    threshold = theta * max_off

    keep = (Aj == row_inds) | (absAx >= threshold[row_inds])
    
    # 构造 S 的 CSR 数组（只保留满足条件的元素）
    Sj_new = Aj[keep]
    Sx_new = Ax[keep]
    # 计算每行保留的元素数量（由于 A 本身是按行存储的）
    row_keep = bm.add.reduceat(keep.astype(int), Ap[:-1])
    Sp_new = bm.concatenate(([0], bm.cumsum(row_keep)))

    # 构造 CSR 格式的强连接矩阵 S
    S = csr_matrix((Sx_new, Sj_new, Sp_new), shape=(n_row, n_row))

    # 计算 C/F 划分（这里调用你已有的 rs_cf_splitting 函数）
    import time
    start = time.time()
    splitting = rs_cf_splitting(S)
    end = time.time()
    print("rs_cf_splitting time:", end - start)
    return splitting, S

def ruge_stuben_coarsen(A, theta=0.025):
    
    """Ruge-Stuben coarsening method for multigrid preconditioning.
    
    This method applies the Ruge-Stuben coarsening technique for multigrid methods.
    It constructs the coarse grid operator by selecting a set of coarse nodes based 
    on the given matrix A. The method reduces the problem size by creating a set of 
    coarse variables (the C-set) and uses a interpolation operator (Pro) to map from 
    the fine grid to the coarse grid.

    Parameters:
        A (CSRMatrix): The input matrix representing the linear system. It should be a sparse matrix.
        theta (float, optional): A threshold parameter used to delete weak connections in the matrix. Default is 0.025.

    Returns:
        Pro (CSRMatrix): The interpolation operator that maps from the fine grid to the coarse grid.
        Res (CSRMatrix): The restriction operator that maps from the coarse grid to the fine grid.

        
    Notes:
        The method assumes that `A` is a M sparse matrix and uses it to construct a coarse grid.
        The interpolation and restriction operators are constructed using the Ruge-Stuben approach.
    """
    
    N = A.shape[0]
    maxaij = A.col_min()+0.05
    inverse_maxaij = 1 / bm.abs(maxaij)
    D = spdiags(inverse_maxaij,diags=0,M = N,N =N)
    Am = D @ A

    # Delete weak connectness
    im, jm, sm = Am.find()
    idx = (-sm > theta)
    As = csr_matrix((bm.ones_like(sm[idx]), (im[idx], jm[idx])), shape=(N, N))
    Am = csr_matrix((sm[idx], (im[idx], jm[idx])), shape=(N, N))
    Ass = As + Am

    isF = bm.zeros(N, dtype=bool)
    degIn = bm.array(As.sum(axis=1)).flatten()
    isF[degIn == 0] = True

    # Find an approximate maximal independent set and put to C set
    isC = bm.zeros(N, dtype=bool)
    U = bm.arange(N)
    degFin = bm.zeros(N)
    while bm.sum(isC) < N / 2 and len(U) > 20:
        isS = bm.zeros(N, dtype=bool)
        degInAll = degIn + degFin
        isS[(bm.random.rand(N) < 0.85 * degInAll / bm.mean(degInAll)) & (degInAll > 0)] = True
        S = bm.where(isS)[0]
        
        i, j, _ = Ass[S,S].triu(k=1).find()
        
        idx = degInAll[S[i]] >= degInAll[S[j]]
        isS[S[j[idx]]] = False
        isS[S[i[~idx]]] = False
        isC[isS] = True

        C = bm.where(isC)[0]
        i ,_,_= Ass[:, C].find()
        
        
        isF[i] = True
        U = bm.where(~(isF | isC))[0]

        degIn[isF | isC] = 0
        degFin = bm.zeros(N)
        
        F = bm.where(isF)[0]
        if U.shape[0] == 0:
            degFin = degFin
        else:
            degFin[U] = (As[F, U]).sum(axis=1)
        
        if len(U) <= 20:
            isC[U] = True
            U = []
    return isC,Am


def ruge_stuben_chen_coarsen(A, theta=0.025):
    """
    @brief Long Chen 修改过的 Ruge-Stuben 粗化方法

    @param[in] A 对称正定矩阵
    @param[in] theta 粗化阈值
    """

    # 1. 初始化参数
    N = A.shape[0]
    isC = bm.zeros(N, dtype=bm.bool)
    N0 = min(int(bm.floor(bm.sqrt(N))), 25)

    # 2. 生成强连通矩阵 
    # 然后函数计算出归一化的矩阵Am（矩阵A的对角线被归一化），
    # 并找出强连接的节点，也就是那些Am的元素值小于阈值theta的节点。
    # 得到的结果保存在矩阵G中。
    Dinv = spdiags(1./bm.sqrt(A.diags().values),diags=0,M = N,N =N)
    
    Am = Dinv @ A @ Dinv # 对角线归一化矩阵
    im, jm, sm = Am.find()
    flag = (-sm > theta) 
    # 删除对角、非对角弱联接项，注意对角线元素为1，也会被过滤掉
    G = csr_matrix((sm[flag], (im[flag], jm[flag])), shape=(N, N))
    isC = rs_cf_splitting(G)
    # # 3. 计算顶点的度 
    # # 函数计算出每个节点的度，也就是与每个节点强连接的节点数量。
    # # 如果有太多的节点没有连接，函数会随机选择N0个节点作为粗糙节点并返回。
    # deg = bm.tensor(G.astype(bm.bool).sum(axis=0).flat,dtype=bm.float64)
    # # deg = bm.tensor(bm.sum(csr_matrix(G, dtype=bm.bool), axis=1).flat,
    # #         dtype=bm.float64)
    
    # if bm.sum(deg > 0) < 0.25*bm.sqrt(N):
    #     isC[bm.random.choice(range(N), N0)] = True
    #     return isC, G

    # flag = (deg > 0)
    # deg[flag] += 0.1 * bm.random.rand(bm.sum(flag))

    # # 4. 寻找最大独立集 
    # # 函数尝试找出一个近似的最大独立集并将其节点添加到粗糙节点集合中。
    # # 如果某节点被标记为粗糙节点，则其相邻的节点会被标记为细节点。
    # isF = bm.zeros(N, dtype=bm.bool)
    # isF[deg == 0] = True # 孤立点为细节点 
    # isU = bm.ones(N, dtype=bm.bool) # 未决定的集合

    # while bm.sum(isC) < N/2 and bm.sum(isU) > N0:
    #     # 如果粗节点的个数少于总节点个数的一半，并且未决定的点集大于 N0
    #     isS = bm.zeros(N, dtype=bm.bool) # 选择集
    #     isS[deg>0] = True # 从非孤立点选择
    #     S = bm.nonzero(isS)[0]
    #     # 非孤立点集的连接关系
    #     #i, j = sp.triu(G[S, :][:, S], 1).nonzero()
    #     i, j = G[S, S].triu(1).nonzero_slice
    #     # 第 i 个非孤立点的度大于等于第 j 个非孤立点的度
    #     flag = deg[S[i]] >= deg[S[j]]
    #     isS[S[j[flag]]] = False # 把度小的节点从选择集移除
    #     isS[S[i[~flag]]] = False # 把度小的节点从选择集移除
    #     isC[isS] = True # 剩下的点就是粗点
    #     C = bm.nonzero(isC)[0]
    #     # Remove coarse nodes and neighboring nodes from undecided set
    #     i, _, _ = G[:, C].find()
    #     isF[i] = True # 粗点的相邻点是细点
    #     isU = ~(isF | isC) # 不是细点也不是粗点，就是未决定点
    #     deg[~isU] = 0 # 粗点或细节的度设置为 0

    #     if bm.sum(isU) <= N0:
    #         # 如果未决定点的数量小于等于 N0，把未决定点设为粗点
    #         isC[isU] = True
    #         isU = []

    return isC, G

def ruge_stuben_chen_coarsen(A, theta=0.025):
    """
    @brief Modified Ruge-Stuben coarsening method by Long Chen

    @param[in] A A symmetric positive definite matrix
    @param[in] theta Coarsening threshold
    """

    # 1. Initialize parameters
    N = A.shape[0]
    isC = bm.zeros(N, dtype=bm.bool)
    N0 = min(int(bm.floor(bm.sqrt(N))), 25)

    # 2. Generate the strong connectivity matrix
    # The function first computes a normalized matrix Am (where the diagonal of A is normalized),
    # and identifies strongly connected nodes, i.e., nodes whose elements in Am are smaller than the threshold theta.
    # The resulting strong connection matrix is stored in G.
    Dinv = spdiags(1./bm.sqrt(A.diags().values), diags=0, M=N, N=N)

    Am = Dinv @ A @ Dinv  # Diagonal-normalized matrix
    im, jm, sm = Am.find()
    flag = (-sm > theta)  
    # Remove weak connections, including diagonal elements (diagonal elements are 1 and will be filtered out)
    G = csr_matrix((sm[flag], (im[flag], jm[flag])), shape=(N, N))

    # 3. Compute vertex degree
    # The function calculates the degree of each node, which is the number of strongly connected nodes.
    # If too many nodes are unconnected, it randomly selects N0 nodes as coarse nodes and returns.
    deg = bm.tensor(G.astype(bm.bool).sum(axis=0).flat, dtype=bm.float64)

    if bm.sum(deg > 0) < 0.25 * bm.sqrt(N):
        isC[bm.random.choice(range(N), N0)] = True
        return isC, G

    flag = (deg > 0)
    deg[flag] += 0.1 * bm.random.rand(bm.sum(flag))

    # 4. Find the maximal independent set
    # The function attempts to find an approximate maximal independent set and adds its nodes to the coarse node set.
    # If a node is marked as a coarse node, its neighboring nodes are marked as fine nodes.
    isF = bm.zeros(N, dtype=bm.bool)
    isF[deg == 0] = True  # Isolated points are fine nodes
    isU = bm.ones(N, dtype=bm.bool)  # Undecided nodes

    while bm.sum(isC) < N/2 and bm.sum(isU) > N0:
        # If the number of coarse nodes is less than half of the total nodes and the number of undecided nodes is greater than N0
        isS = bm.zeros(N, dtype=bm.bool)  # Selection set
        isS[deg > 0] = True  # Select from non-isolated nodes
        S = bm.nonzero(isS)[0]
        # Connectivity of the non-isolated node set
        i, j = G[S, S].triu(1).nonzero_slice
        # Check if the degree of node i is greater than or equal to that of node j
        flag = deg[S[i]] >= deg[S[j]]
        isS[S[j[flag]]] = False  # Remove lower-degree nodes from the selection set
        isS[S[i[~flag]]] = False  # Remove lower-degree nodes from the selection set
        isC[isS] = True  # The remaining nodes are coarse nodes
        C = bm.nonzero(isC)[0]
        # Remove coarse nodes and their neighbors from the undecided set
        i, _, _ = G[:, C].find()
        isF[i] = True  # Neighboring nodes of coarse nodes are fine nodes
        isU = ~(isF | isC)  # Nodes that are neither fine nor coarse remain undecided
        deg[~isU] = 0  # Set the degree of coarse and fine nodes to 0

        if bm.sum(isU) <= N0:
            # If the number of undecided nodes is less than or equal to N0, set them as coarse nodes
            isC[isU] = True
            isU = []

    return isC, G

