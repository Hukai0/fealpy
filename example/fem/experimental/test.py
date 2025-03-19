import numpy as np

# 示例 lambda_vals 数据
lambda_vals = np.array([3, 1, 2, 2, 4])

# 实际代码
def actual_code(lambda_vals):
    # 对 lambda_vals 排序并获取排序后的索引



        # 对 lambda_vals 排序并获取排序后的索引
    sorted_indices = np.argsort(lambda_vals)
    
    # 将索引值填充到 node_to_index
    node_to_index = np.arange(len(lambda_vals))
    node_to_index[sorted_indices] = np.arange(len(lambda_vals))
    
    # 逆映射
    index_to_node = np.argsort(node_to_index)
    


    
    return node_to_index, index_to_node

# 注释代码
def comment_code(lambda_vals):
    n_nodes = len(lambda_vals)
    
    # interval_ptr 和 interval_count 假设
    interval_ptr = np.zeros(n_nodes, dtype=int)
    interval_count = np.zeros(n_nodes, dtype=int)
    
    # 生成基于 lambda_vals 排序的索引
    sorted_indices = np.argsort(lambda_vals)
    node_to_index = np.empty_like(sorted_indices)
    index_to_node = np.zeros(n_nodes, dtype=int)
    
    # 使用 interval_ptr 和 interval_count 更新 index_to_node 和 node_to_index
    for i in range(n_nodes):
        lambda_i = lambda_vals[i]
        index = interval_ptr[lambda_i] + interval_count[lambda_i]
        index_to_node[index] = i
        node_to_index[i] = index
        interval_count[lambda_i] += 1
    
    return sorted_indices, node_to_index, index_to_node

# 测试程序
def test_program():
    node_to_index, index_to_node_sorted = actual_code(lambda_vals)
    print("实际代码结果:")
    print(f"node_to_index: {node_to_index}")
    print(f"index_to_node_sorted (逆映射): {index_to_node_sorted}")

    sorted_indices_c, node_to_index_c, index_to_node_c = comment_code(lambda_vals)
    print("\n注释代码结果:")
    print(f"sorted_indices: {sorted_indices_c}")
    print(f"node_to_index: {node_to_index_c}")
    print(f"index_to_node: {index_to_node_c}")

# 执行测试程序
test_program()
