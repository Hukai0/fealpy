import numpy as np
from scipy import sparse

def standard_aggregation(C):

    n_row = C.shape[0]
    Ap = C.indptr
    Aj = C.indices

    # 初始化聚集标记数组 x 和 coarse points 数组 y
    x = np.zeros(n_row, dtype=int)   # 0 表示未聚集
    y = np.empty(n_row, dtype=int)     # 预分配空间，后续只使用前 next_aggregate 个元素

    # 聚集编号从 1 开始（最后调整为从 0 开始）
    next_aggregate = 1

    # ----- Pass 1: 对每个节点，若其邻居均未被聚集，则把它及其邻居归为一个新聚集 -----
    for i in range(n_row):
        if x[i] != 0:
            continue  # 已聚集，跳过

        # 取 i 行的邻居
        row_start, row_end = Ap[i], Ap[i+1]
        neighbors = Aj[row_start:row_end]
        # 排除自环：即 neighbors != i
        non_self = neighbors[neighbors != i]
        has_neighbors = (non_self.size > 0)

        # 检查非自身邻居中是否有已经聚集的
        has_aggregated_neighbors = False
        if has_neighbors:
            has_aggregated_neighbors = np.any(x[non_self] != 0)

        if not has_neighbors:
            # 没有邻居：标记为孤立节点，使用 -n_row 作为特殊标记
            x[i] = -n_row
        elif not has_aggregated_neighbors:
            # 没有已有聚集的邻居：以 i 为种子形成新聚集
            x[i] = next_aggregate
            y[next_aggregate-1] = i   # 记录聚集代表
            # 将 i 的所有邻居（包括 i 自己）归入此聚集
            x[neighbors] = next_aggregate
            next_aggregate += 1

    # ----- Pass 2: 对仍未聚集的节点，将它们加入任一已有聚集 -----
    for i in range(n_row):
        if x[i] != 0:
            continue
        # 对 i 行所有邻居
        row_start, row_end = Ap[i], Ap[i+1]
        neighbors = Aj[row_start:row_end]
        for j in neighbors:
            if x[j] > 0:
                x[i] = -x[j]  # 用负值标记，表示 i 属于 j 所在聚集
                break

    # 将 next_aggregate 减 1 得到聚集总数
    num_aggregates = next_aggregate - 1 

    # ----- Pass 3: 调整聚集编号
    for i in range(n_row):
        xi = x[i]
        if xi != 0:
            if xi > 0:
                x[i] = xi - 1        # 种子节点：编号减 1
            elif xi == -n_row:
                x[i] = -1             # 孤立节点统一标记为 -1
            else:
                x[i] = -xi -1       # 非种子：取正值后减 1
        else:
            # 对于还未聚集的节点，单独形成一个新聚集
            row_start, row_end = Ap[i], Ap[i+1]
            neighbors = Aj[row_start:row_end]
            x[i] = num_aggregates  # 使用当前 num_aggregates 作为新聚集编号
            y[num_aggregates] = i   # 新聚集的代表
            # 将所有未聚集的邻居归入此聚集
            unmarked = neighbors[x[neighbors] == 0]
            x[unmarked] = num_aggregates
            num_aggregates += 1
    print("x:", x)
    # ----- 构造聚集算子 AggOp -----
    # 这里每个节点对应一行，聚集编号存储在 x 中。
    # 对于孤立节点，x[i] == -1，我们将其归为聚集 0。
    agg_col = np.where(x == -1,num_aggregates , x)
    data = np.ones(n_row, dtype=np.int32)
    row = np.arange(n_row, dtype=Ap.dtype)
    AggOp = sparse.csr_matrix((data, (row, agg_col)), shape=(n_row, num_aggregates))

    # 截取 coarse points 数组，只保留前 next_aggregate 个元素
    Cpts = y[:num_aggregates]

    return AggOp, Cpts

# 示例测试
if __name__ == '__main__':
    # 构造一个简单图的 CSR 表示（6 个节点）
    # 邻接关系（无向对称）：
    # 0: 1,2
    # 1: 0,2,3
    # 2: 0,1,3
    # 3: 1,2,4,5
    # 4: 3,5
    # 5: 3,4
    #
    # CSR 数组：
    Ap = np.array([0, 2, 5, 8, 12, 14, 16])
    Aj = np.array([1,2, 0,2,3, 0,1,3, 1,2,4,5, 3,5, 3,4])
    # 这里 A 的数值对于聚集计算不重要，只需要非零结构；用全 1 即可
    Ax = np.ones(len(Aj), dtype=float)

    # 构造 CSR 矩阵 C
    C = sparse.csr_matrix((Ax, Aj, Ap), shape=(6, 6))

    AggOp, Cpts = standard_aggregation(C)
    print("Aggregation operator AggOp (dense format):")
    print(AggOp.toarray())
    print("Coarse points (Cpts):")
    print(Cpts)

import numpy as np
from scipy.sparse import csr_matrix, issparse, bsr_array


def fit_candidates(AggOp, B, tol=1e-10):

    
    B = np.asarray(B)
    # Check compatibility: number of fine blocks = AggOp.shape[0]
    N_fine, N_coarse = AggOp.shape
    if B.shape[0] % N_fine != 0:
        raise ValueError(f"Incompatible dimensions: AggOp.shape={AggOp.shape}, B.shape={B.shape}")
    K1 = B.shape[0] // N_fine   # fine-level block size
    K2 = B.shape[1]             # number of candidate vectors

    # Allocate coarse candidate matrix R: initially shape (N_coarse, K2, K2)
    R = np.zeros((N_coarse, K2, K2), dtype=B.dtype)
    # Allocate Qx: will store the tentative prolongator blocks.
    # Qx has shape (AggOp.nnz, K1, K2), one block per nonzero of AggOp.
    Qx = np.empty((AggOp.nnz, K1, K2), dtype=B.dtype)

    # Convert AggOp to CSC for efficient column access.
    AggOp_csc = AggOp.tocsc()
    
    # --- Step 1: Copy blocks from B into Qx according to AggOp ---
    # AggOp_csc.indptr and AggOp_csc.indices describe, for each coarse block (column),
    # the fine-level block indices that belong to that aggregate.
    index = 0
    # 遍历每 coarse block (聚集)
    for j in range(N_coarse):
        col_start = AggOp_csc.indptr[j]
        col_end = AggOp_csc.indptr[j+1]
        # 对于 coarse block j，下属的 fine-level blocks的数量：
        for ii in range(col_start, col_end):
            block_idx = AggOp_csc.indices[ii]  # 细级块号
            # 复制 B 中对应块的数据：B 的块在行区间 [block_idx*K1, (block_idx+1)*K1)
            Qx[index, :, :] = B[block_idx*K1:(block_idx+1)*K1, :]
            index += 1

    # --- Step 2: Orthonormalize columns for each coarse block ---
    # 对于每个 coarse block j，提取其所有块组成的矩阵 X of shape (nblocks*K1, K2)
    for j in range(N_coarse):
        start = AggOp_csc.indptr[j]
        end = AggOp_csc.indptr[j+1]
        nblocks = end - start
        if nblocks == 0:
            continue  # 如果该聚集中没有任何 fine-level块，则跳过
        # X: 将这些块堆叠成一个大矩阵，形状 (nblocks*K1, K2)
        X = Qx[start:end, :, :].reshape(nblocks*K1, K2)
        # 对每个 candidate column bj 进行正交化
        for bj in range(K2):
            # 计算当前列的初始范数
            norm_j = np.linalg.norm(X[:, bj])
            threshold_j = tol * norm_j
            # 对当前列 bj 与之前的每一列 bi 进行正交化
            for bi in range(bj):
                dot_prod = np.dot(X[:, bj], X[:, bi])
                X[:, bj] -= dot_prod * X[:, bi]
                R[j, bi, bj] = dot_prod
            # 重新计算归一化后的范数
            norm_j = np.linalg.norm(X[:, bj])
            if norm_j > threshold_j:
                scale = 1.0 / norm_j
                R[j, bj, bj] = norm_j
            else:
                scale = 0.0
                R[j, bj, bj] = 0.0
            X[:, bj] *= scale
        # 将正交化后的 X 写回 Qx
        Qx[start:end, :, :] = X.reshape(nblocks, K1, K2)

    # --- Step 3: Reshape coarse candidate matrix R ---
    # 将 R 从形状 (N_coarse, K2, K2) 变换为 (N_coarse*K2, K2)
    R = R.reshape(N_coarse*K2, K2)

    # --- Step 4: Construct the tentative prolongator Q in BSR format ---
    # 根据 AggOp_csc.indices 和 AggOp_csc.indptr，将 Qx 作为数据数组构造 BSR 矩阵
    # Qx 的形状为 (AggOp.nnz, K1, K2)，我们需要将块中的数据转置以匹配期望的尺寸。
    # 构造 BSR 矩阵 Q 的初步形状为 (K2*N_coarse, K1*N_fine)
    Q = bsr_array((Qx.swapaxes(1, 2).copy(), AggOp_csc.indices, AggOp_csc.indptr),
                  shape=(K2*N_coarse, K1*N_fine))
    # 最后转置并转换为 BSR 格式，使得 Q 的形状变为 (K1*N_fine, K2*N_coarse)
    Q = Q.T.tobsr()

    return Q, R


# 示例测试
if __name__ == '__main__':
    # 示例：四个节点分成两个聚集
    # 假设 AggOp 描述了 4 个细级块分为两个聚集，其稀疏结构如下：
    # 第一聚集包含细级块 0 和 1，第二聚集包含细级块 2 和 3。
    from scipy.sparse import csr_matrix
    AggOp = csr_matrix([[1, 0],
                        [1, 0],
                        [0, 1],
                        [0, 1]])
    # B: 细级近零空间候选向量，假设每个细级块的尺寸 K1 = 1，
    # 并且候选向量数 K2 = 1，即 B 的形状为 (4, 1)
    B = np.ones((4, 2), dtype=float)
    tol = 1e-10
    Q, R = fit_candidates(AggOp, B, tol)
    print("Tentative prolongator Q (dense):")
    print(Q.toarray())
    print("Coarse-level candidate matrix R:")
    print(R)

