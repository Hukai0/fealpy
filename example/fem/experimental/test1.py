import numpy as np

# 随机生成 5 个点
points = np.random.rand(5, 2)  # (5, 2) 形状

# 传统方式：使用双重 for 循环（慢）
N = len(points)
dist_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
print(dist_matrix)
# 数组化方式：扩展维度 + 广播（快）
p1 = points[:, np.newaxis, :]  # 变成 (5, 1, 2)，空出一个轴
p2 = points[np.newaxis, :, :]  # 变成 (1, 5, 2)

print(p1)
print(p2)

dist_matrix = np.linalg.norm(p1 - p2, axis=2)  # 形状变为 (5, 5)

print(dist_matrix)

from line_profiler import LineProfiler

def test_function():
    for _ in range(1000):
        sum(range(1000))

lp = LineProfiler()
lp.add_function(test_function)
lp.enable()
test_function()
lp.disable()
lp.print_stats()
