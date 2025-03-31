from fealpy.backend import backend_manager as bm
from fealpy.sparse.ops import spdiags
from fealpy.sparse import csr_matrix

def rs_direct_interpolation(Ap, Aj, Ax, Sp, Sj, Sx, splitting):

    n_nodes = len(splitting)
    row_indices = bm.repeat(bm.arange(n_nodes), Sp[1:] - Sp[:-1])     # 2. 判断每个邻接记录是否满足条件：对应的 Sj 是 C 点，且不等于自身
    valid = splitting[Sj] & (Sj != row_indices)  # 3. 对每个行号统计符合条件的邻接记录个数
    neighbor_counts = bm.bincount(row_indices, weights=valid.astype(int), minlength=n_nodes)   # 4. 对于 C 点固定计数为 1，F 点使用统计到的邻接记录个数
    counts = bm.where(splitting, 1, neighbor_counts)     # 5. 构造 P 的行指针（CSR格式），首元素为 0，后续为累计和
    Pp = bm.concatenate(([0], bm.cumsum(counts).astype(int)))   # 计算 P 的列索引 (Pj) 和数值 (Px)
    Pj = bm.zeros(Pp[-1], dtype=int)
    Px = bm.zeros(Pp[-1], dtype=float)

    for i in range(n_nodes):
        if splitting[i] == True:
            Pj[Pp[i]] = i
            Px[Pp[i]] = 1.0  # C 点直接映射到自身
        else:      # 计算 F 点插值的权重
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

            alpha = sum_all_neg / sum_strong_neg if sum_strong_neg != 0 else 0
            beta = sum_all_pos / sum_strong_pos if sum_strong_pos != 0 else 0

            if sum_strong_pos == 0:
                diag += sum_all_pos
                beta = 0

            neg_coeff = -alpha / diag if diag != 0 else 0
            pos_coeff = -beta / diag if diag != 0 else 0         # 计算 P 的插值系数
            nnz_index = Pp[i]
            for jj in range(Sp[i], Sp[i+1]):
                if splitting[Sj[jj]] == True and Sj[jj] != i:
                    Pj[nnz_index] = Sj[jj]
                    Px[nnz_index] = neg_coeff * Sx[jj] if Sx[jj] < 0 else pos_coeff * Sx[jj]
                    nnz_index += 1

    coarse_map = bm.cumsum(splitting) - 1  # 计算 C 点编号映射
    Pj[:] = coarse_map[Pj[:]]
    n_coarse = bm.sum(splitting)  # C 点个数
    p = csr_matrix((Px, Pj, Pp), shape=(n_nodes, n_coarse))
    r = p.T
    return p, r

def rs_direct_interpolation(Ap, Aj, Ax, Sp, Sj, Sx, splitting):
    n_nodes = len(splitting)
    splitting = bm.asarray(splitting, dtype=bool)
    # 1. 构造 P 的行指针 Pp
    row_indices = bm.repeat(bm.arange(n_nodes), Sp[1:] - Sp[:-1])
    valid = splitting[Sj] & (Sj != row_indices)
    neighbor_counts = bm.bincount(row_indices, weights=valid.astype(int), minlength=n_nodes)
    counts = bm.where(splitting, 1, neighbor_counts)
    Pp = bm.concatenate(([0], bm.cumsum(counts).astype(int)))

    # 2. 初始化 Pj 和 Px
    Pj = bm.empty(Pp[-1], dtype=int)
    Px = bm.empty(Pp[-1], dtype=float)

    # 3. 处理 C 点（向量化赋值）

    C_indices = bm.arange(n_nodes)[splitting]
    Pj[Pp[C_indices]] = C_indices
    Px[Pp[C_indices]] = 1.0
    
    F_indices = bm.arange(n_nodes)[~splitting]

    # 预计算强连接相关的和

    jj_ranges = [bm.arange(Sp[i], Sp[i+1]) for i in F_indices]
    jj_all = bm.concatenate(jj_ranges)
    i_all = bm.repeat(F_indices, [len(r) for r in jj_ranges])
    valid_strong = splitting[Sj[jj_all]] & (Sj[jj_all] != i_all)
    Sx_strong = Sx[jj_all]
    sum_strong_pos = bm.bincount(i_all, weights=bm.where(valid_strong & (Sx_strong >= 0), Sx_strong, 0), minlength=n_nodes)
    sum_strong_neg = bm.bincount(i_all, weights=bm.where(valid_strong & (Sx_strong < 0), Sx_strong, 0), minlength=n_nodes)
 


    # 预计算所有连接相关的和及对角线值
    jj_ranges_A = [bm.arange(Ap[i], Ap[i+1]) for i in F_indices]
    jj_all_A = bm.concatenate(jj_ranges_A)
    i_all_A = bm.repeat(F_indices, [len(r) for r in jj_ranges_A])
    Ax_all = Ax[jj_all_A]
    is_diag = Aj[jj_all_A] == i_all_A
    diag = bm.bincount(i_all_A, weights=bm.where(is_diag, Ax_all, 0), minlength=n_nodes)
    sum_all_pos = bm.bincount(i_all_A, weights=bm.where(~is_diag & (Ax_all >= 0), Ax_all, 0), minlength=n_nodes)
    sum_all_neg = bm.bincount(i_all_A, weights=bm.where(~is_diag & (Ax_all < 0), Ax_all, 0), minlength=n_nodes)

    # 5. 计算 alpha 和 beta
    alpha = bm.where(sum_strong_neg != 0, sum_all_neg / bm.where(sum_strong_neg == 0, 1, sum_strong_neg), 0)
    beta = bm.where(sum_strong_pos != 0, sum_all_pos / bm.where(sum_strong_pos == 0, 1, sum_strong_pos), 0)

    # 6. 计算插值系数
    neg_coeff = bm.where(diag != 0, -alpha / bm.where(diag == 0, 1, diag), 0)
    pos_coeff = bm.where(diag != 0, -beta / bm.where(diag == 0, 1, diag), 0)
    
    # 7. 构造 F 点的 Pj 和 Px
    for i in F_indices:
        start, end = Pp[i], Pp[i+1]
        jj_strong = bm.arange(Sp[i], Sp[i+1])
        valid_jj = splitting[Sj[jj_strong]] & (Sj[jj_strong] != i)
        Pj[start:end] = Sj[jj_strong][valid_jj]
        Sx_jj = Sx[jj_strong][valid_jj]
        Px[start:end] = bm.where(Sx_jj < 0, neg_coeff[i] * Sx_jj, pos_coeff[i] * Sx_jj)

    # 8. 映射到粗网格并构造稀疏矩阵
    coarse_map = bm.cumsum(splitting) - 1
    Pj[:] = coarse_map[Pj[:]]
    n_coarse = bm.sum(splitting)
    p = csr_matrix((Px, Pj, Pp), shape=(n_nodes, n_coarse))
    r = p.T
    return p, r

def rs_standard_interpolation(A, S, splitting):
    A = A.to_scipy()
    S = S.to_scipy()
    Ap, Aj, Ax = A.indptr, A.indices, A.data
    Sp, Sj, Sx = S.indptr, S.indices, S.data
    n_nodes = len(splitting)
    #首先为每一个F点i收集所有C邻居
    F_points = [i for i in range(n_nodes) if not splitting[i]]
    C_points = [i for i in range(n_nodes) if splitting[i]]
    strong_F_neighbors = [[] for _ in range(n_nodes)]
    all_C_neighbors = [[] for _ in range(n_nodes)]
    for i in F_points:
        for jj in range(Sp[i], Sp[i+1]):
            j = Sj[jj]
            if not splitting[j] and j != i:  # j是强F邻点
                strong_F_neighbors[i].append(j)
    # === 修改：基于扩展C点集合和修正后的hat_A计算插值 ===
    extended_C = [[] for _ in range(n_nodes)]
    for i in F_points:
        # 直接C邻点
        direct_C = [Sj[jj] for jj in range(Sp[i], Sp[i+1]) 
                    if splitting[Sj[jj]] and Sj[jj] != i]
        # 强F邻点的C邻点
        indirect_C = []
        for j in strong_F_neighbors[i]:
            indirect_C += [Sj[kk] for kk in range(Sp[j], Sp[j+1]) 
                         if splitting[Sj[kk]] and Sj[kk] != j]
        # 合并并去重
        extended_C[i] = list(set(direct_C + indirect_C))
    for i in F_points:
        F_C = [Sj[jj] for jj in range(Sp[i], Sp[i+1]) if splitting[Sj[jj]] and Sj[jj] != i]
        F_F = []
        for j in strong_F_neighbors[i]:
            F_F += [Sj[kk] for kk in range(Sp[j], Sp[j+1]) if splitting[Sj[kk]] and Sj[kk]!= j]
        all_C_neighbors[i] = list(set(F_C + F_F))
    #构造P的行指针
    counts = []
    for i in range(n_nodes):
        if splitting[i]:
            counts.append(1)#C只占1个位置
        else:
            counts.append(len(all_C_neighbors[i]))
    #修改hat_A
    hat_A = A.copy()
    for i in F_points:
        for j in strong_F_neighbors[i]:
            a_ij = A[i,j]
            a_ji = A[j,i]
            a_jj = A[j,j]
            if a_jj == 0:
                continue
            hat_A[i,i] -= (a_ij*a_ji) / a_jj
            for kk in range(Sp[j], Sp[j+1]):
                k = Sj[kk]
                if splitting[k]:
                    a_jk = A[j,k]
                    hat_A[i,k] -= (a_ij * a_jk) / a_jj 
    hat_Ap, hat_Aj, hat_Ax = hat_A.indptr, hat_A.indices, hat_A.data
    Pp = bm.concatenate(([0], bm.cumsum(counts).astype(int)))
    Pj = bm.zeros(Pp[-1], dtype=int)
    Px = bm.zeros(Pp[-1], dtype=float)
    for i in range(n_nodes):
        if splitting[i]:
            Pj[Pp[i]] = i
            Px[Pp[i]] = 1.0
        else:
            C_neighbors = all_C_neighbors[i]
            if not C_neighbors:
                continue
            sum_strong_neg = 0.0
            sum_strong_pos = 0.0
            for k in C_neighbors:
                hat_A_ik = hat_A[i,k]
                if hat_A_ik < 0:
                    sum_strong_neg += hat_A_ik
                elif hat_A_ik > 0:
                    sum_strong_pos += hat_A_ik
            sum_neg = sum(hat_Ax[jj] for jj in range(hat_Ap[i], hat_Ap[i+1]) if hat_Ax[jj] < 0 and hat_Aj[jj] != i)
            sum_pos = sum(hat_Ax[jj] for jj in range(hat_Ap[i], hat_Ap[i+1]) if hat_Ax[jj] > 0 and hat_Aj[jj] != i)
            alpha = sum_neg / sum_strong_neg if sum_strong_neg != 0 else 0
            beta = sum_pos / sum_strong_pos if sum_strong_pos != 0 else 0
            diag = hat_A[i,i]
            if sum_strong_pos == 0:
                diag += sum_pos
                beta = 0
            # 计算插值权重
            neg_coeff = -alpha / diag if diag != 0 else 0
            pos_coeff = -beta / diag if diag != 0 else 0
            ptr = Pp[i]
            for k in C_neighbors:
                hat_A_ik = hat_A[i,k]
                if hat_A_ik < 0:
                    Px[ptr] = neg_coeff * hat_A_ik
                elif hat_A_ik > 0:
                    Px[ptr] = pos_coeff * hat_A_ik
                Pj[ptr] = k
                ptr += 1
    coarse_map = bm.cumsum(splitting) - 1
    Pj[:] = coarse_map[Pj]
    n_coarse = bm.sum(splitting)
    P = csr_matrix((Px, Pj, Pp), shape=(n_nodes, n_coarse))
    R = P.T
    return P,R

def ruge_stuben_interpolation(Am,isC):
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
    
    ti, tj= (Dsum @ Afc).nonzero_slice
    tw = (Dsum @ Afc).data

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

def two_points_interpolation(A, isC):
    """
    @brief 构造延长和限止矩阵，第一个细节点最多用两个粗节点 
    """
    N = A.shape[0]

    # 1. 索引映射：函数首先创建了一个从粗网格到细网格的索引映射。它找出了所
    #    有的粗节点和细节点，并存储了他们的索引。
    allNode = bm.arange(N)
    fineNode = allNode[~isC]
    HB = bm.zeros((len(fineNode), 3))
    HB[:, 0] = fineNode
    NC = N - len(fineNode)
    coarseNode = bm.arange(NC)
    coarseNodeFineIdx = bm.nonzero(isC)[0]

    # 2. 构造矩阵依赖的延长和限止算子
    Afc = A[fineNode, coarseNodeFineIdx]
    i, j, s = Afc.find()
    HB[i[::-1], 2] = j[::-1]
    HB[i, 1] = j
    w = bm.zeros((len(fineNode), 2))
    w[i[::-1], 1] = s[::-1]
    w[i, 0] = s
    Dsum =bm.sum(w, axis=1)
    flag = Dsum != 0
    w = w[flag, :] / Dsum[flag, None]
    HB = HB[flag, :]
    I = bm.concatenate((coarseNodeFineIdx, HB[:, 0], HB[:, 0]))
    J = bm.concatenate((coarseNode, HB[:, 1], HB[:, 2]))
    val = bm.concatenate((bm.ones(NC), w[:, 0], w[:, 1]))
    print(I.max(), J.max(), val.max(), I.shape, J.shape, val.shape,N, NC)
    P = csr_matrix((val, (I.astype(int), J.astype(int))), shape=(N, NC))
    R = P.T

    return P, R