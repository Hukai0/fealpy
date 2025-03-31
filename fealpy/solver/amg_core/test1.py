import numpy as np
import scipy.sparse as sp
from fealpy.backend import backend_manager as bm
#from fealpy.sparse import csr_matrix, bsr_array

def fit_candidates(AggOp, B, tol=1e-10):

    
    N_fine, N_coarse = AggOp.shape
    # B 的行数必须等于 N_fine
    if B.shape[0] != N_fine:
        raise ValueError(f"Incompatible dimensions: AggOp.shape={AggOp.shape}, B.shape={B.shape}")
    
    K2 = B.shape[1]   # 候选向量个数

    # 分配 R，初始形状 (N_coarse, K2, K2)
    R = bm.zeros((N_coarse, K2, K2), dtype=B.dtype)
    # 分配 Qx，存储 Tentative Prolongator 的块数据，每个非零条目对应一个块，形状 (AggOp.nnz, 1, K2)
    Qx = bm.empty((AggOp.nnz, K2), dtype=B.dtype)
    
    # 将 AggOp 转换为 CSC 格式便于按聚集处理
    AggOp_csc = AggOp.tocsc()
    
    # --- Step 1: 从 B 中复制数据到 Qx，根据 AggOp 的非零结构 ---
    index = 0
    for j in range(N_coarse):
        col_start = AggOp_csc.indptr[j]
        col_end = AggOp_csc.indptr[j+1]
        for ii in range(col_start, col_end):
            block_idx = AggOp_csc.indices[ii]  # 细级块索引（在这里就是行号，因为 K1=1）
            Qx[index, :] = B[block_idx, :]     # 直接复制该行数据
            index += 1

    # --- Step 2: 对每个粗级块进行正交化 ---
    for j in range(N_coarse):
        start = AggOp_csc.indptr[j]
        end = AggOp_csc.indptr[j+1]
        nblocks = end - start
        if nblocks == 0:
            continue
        # 将这一聚集内所有候选向量堆叠成矩阵 X, 形状 (nblocks, K2)
        X = Qx[start:end, :]
        for bj in range(K2):
            norm_j = bm.linalg.norm(X[:, bj])
            threshold_j = tol * norm_j
            # 对当前列 bj 与之前的每一列进行正交化
            for bi in range(bj):
                dot_prod = bm.dot(X[:, bj], X[:, bi])
                X[:, bj] -= dot_prod * X[:, bi]
                R[j, bi, bj] = dot_prod
            norm_j = bm.linalg.norm(X[:, bj])
            if norm_j > threshold_j:
                scale = 1.0 / norm_j
                R[j, bj, bj] = norm_j
            else:
                scale = 0.0
                R[j, bj, bj] = 0.0
            X[:, bj] *= scale
        Qx[start:end, :] = X

    # --- Step 3: 重塑 R 为二维矩阵 (N_coarse*K2, K2) ---
    R = R.reshape(N_coarse * K2, K2)

    # --- Step 4: 直接构造延拓矩阵 Q 的 CSR 表示 ---
    # 这里每个 Qx 中的行对应一个非零条目，
    # 每个细级块 i 对应全局行号 i，
    # 每个聚集 j 对应全局列号范围 [j*K2, (j+1)*K2)
    row_list = []
    col_list = []
    data_list = []
    for j in range(N_coarse):
        col_start = AggOp_csc.indptr[j]
        col_end = AggOp_csc.indptr[j+1]
        for ii in range(col_start, col_end):
            block_idx = AggOp_csc.indices[ii]
            # 细级块 block_idx 对应全局行号 = block_idx (因为 K1=1)
            global_row = block_idx
            # 对应的候选向量为 Qx[ii, :]，全局列号 = j*K2 到 (j+1)*K2-1
            global_cols = j * K2 + np.arange(K2)
            # 将这行数据复制到结果中
            row_list.append(np.full(K2, global_row, dtype=AggOp_csc.indptr.dtype))
            col_list.append(global_cols)
            data_list.append(Qx[ii, :])
    
    rows = np.concatenate(row_list)
    cols = np.concatenate(col_list)
    data = np.concatenate(data_list)
    shape = (N_fine, N_coarse * K2)
    Q = sp.csr_matrix((data, (rows, cols)), shape=shape)
    
    return Q, R

# 示例测试
if __name__ == '__main__':
    # 构造一个示例 AggOp：假设有 4 个细级块聚到 2 个粗级块
    AggOp = sp.csr_matrix([[1, 0],
                           [1, 0],
                           [0, 1],
                           [0, 1]])
    # 细级候选向量矩阵 B 形状为 (N_fine, K2)，这里 N_fine = 4, K2 = 2
    B = np.array([[1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2]], dtype=float)
    
    Q, R = fit_candidates(AggOp, B, tol=1e-10)
    print("Tentative prolongator Q (dense):")
    print(Q.toarray())
    print("Coarse candidate matrix R:")
    print(R)
    
    import numpy as np
import scipy.sparse as sp
from fealpy.backend import backend_manager as bm

def fit_candidates(AggOp, B, tol=1e-10):

    
    N_fine, N_coarse = AggOp.shape
    # 检查 B 的行数是否等于 N_fine
    if B.shape[0] != N_fine:
        raise ValueError(f"Incompatible dimensions: AggOp.shape={AggOp.shape}, B.shape={B.shape}")
    K2 = B.shape[1]  # 候选向量个数
    
    # Step 1. 按聚集归类
    # 建立字典：对于每个粗级聚集 j，收集所有细级自由度 i 使得 AggOp[i, j] 非零
    agg_dict = {j: [] for j in range(N_coarse)}
    # 遍历每个细级自由度 i
    for i in range(N_fine):
        row_start, row_end = AggOp.indptr[i], AggOp.indptr[i+1]
        if row_end - row_start == 0:
            continue  # 无聚集关系，跳过
        # 对于该行的每个非零位置，取出聚集编号 j
        for idx in range(row_start, row_end):
            j = AggOp.indices[idx]
            agg_dict[j].append(i)
    
    # Step 2. 对每个聚集 j，提取候选向量，进行局部正交化
    # 存储每个聚集得到的正交化结果，以及局部 R
    Qx_blocks = []  # 每个元素：一个矩阵，行数 = number of fine DOFs in aggregate, 列数 = K2
    R_blocks = []   # 每个元素：局部 R，形状 (K2, K2)
    
    for j in range(N_coarse):
        indices = agg_dict[j]
        if len(indices) == 0:
            # 如果某个聚集没有对应细级自由度，则直接设置局部 R 为零矩阵
            R_blocks.append(np.zeros((K2, K2), dtype=B.dtype))
            continue
        # 构造聚集 j 的候选矩阵 X: 每一行来自 B 对应的细级自由度
        X = B[indices, :]  # 形状 (n_j, K2)
        # 对 X 的列进行 Gram–Schmidt 正交化
        n_j = X.shape[0]
        R_local = np.zeros((K2, K2), dtype=B.dtype)
        for bj in range(K2):
            # 初始范数
            norm_j = np.linalg.norm(X[:, bj])
            threshold_j = tol * norm_j
            for bi in range(bj):
                dot_prod = np.dot(X[:, bj], X[:, bi])
                X[:, bj] -= dot_prod * X[:, bi]
                R_local[bi, bj] = dot_prod
            norm_j = np.linalg.norm(X[:, bj])
            if norm_j > threshold_j:
                scale = 1.0 / norm_j
                R_local[bj, bj] = norm_j
            else:
                scale = 0.0
                R_local[bj, bj] = 0.0
            X[:, bj] *= scale
        Qx_blocks.append((indices, X))
        R_blocks.append(R_local)
    
    # Step 3. 构造全局延拓矩阵 Q (CSR格式)
    # 对于每个聚集 j，每个细级自由度 i 属于该聚集对应 Q 的一行，
    # 对应的块（行向量，长度 = K2）填入全局矩阵 Q 中，列位置为 j*K2 ~ (j+1)*K2-1。
    row_list = []
    col_list = []
    data_list = []
    for j in range(N_coarse):
        if j >= len(R_blocks):
            continue
        if j not in agg_dict or len(agg_dict[j]) == 0:
            continue
        indices, X = Qx_blocks[j]
        # 对于聚集 j 中的每个细级自由度 i
        for local_i, i in enumerate(indices):
            # 全局行号就是 i（细级自由度数与行数一致，因为 K1=1）
            row = i
            # Q 的列对应于聚集 j 的位置，范围为 [j*K2, (j+1)*K2)
            cols = j * K2 + np.arange(K2)
            # 对应数据：这一行的候选向量 X[local_i, :]
            row_list.append(np.full(K2, row, dtype=int))
            col_list.append(cols)
            data_list.append(X[local_i, :])
    if len(row_list) > 0:
        rows = np.concatenate(row_list)
        cols = np.concatenate(col_list)
        data = np.concatenate(data_list)
    else:
        rows = np.array([], dtype=int)
        cols = np.array([], dtype=int)
        data = np.array([], dtype=B.dtype)
    Q_shape = (N_fine, N_coarse * K2)
    Q = sp.csr_matrix((data, (rows, cols)), shape=Q_shape)
    
    # Step 4. 构造全局粗级候选矩阵 R
    # R_blocks 每个局部 R 的形状为 (K2, K2)，按聚集 j 顺序堆叠起来
    R_global = np.vstack([R_blocks[j] for j in range(N_coarse) if len(agg_dict[j]) > 0])
    
    return Q, R_global

# 示例测试
if __name__ == '__main__':
    # 构造一个示例 AggOp：假设有 4 个细级自由度聚到 2 个粗级聚集
    AggOp = sp.csr_matrix([[1, 0],
                           [1, 0],
                           [0, 1],
                           [0, 1]])
    # 细级候选向量矩阵 B，形状为 (N_fine, K2) ；假设 N_fine = 4, K2 = 2
    B = np.array([[1, 2],
                  [1, 2],
                  [1, 2],
                  [1, 2]], dtype=float)
    
    Q, R = fit_candidates(AggOp, B, tol=1e-10)
    print("Tentative prolongator Q (dense):")
    print(Q.toarray())
    print("Coarse candidate matrix R:")
    print(R)

