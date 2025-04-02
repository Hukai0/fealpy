import numpy as np

# 构造一阶张量 A，形状 (4,)
A = np.array([1, 2, 3, 4])
print("张量 A =", A, "形状：", A.shape)

# 构造二阶张量 B，形状 (4, 5)
B = np.arange(20).reshape(4, 5)
print("\n张量 B =\n", B, "\n形状：", B.shape)

# 构造三阶张量 C，形状 (4, 5, 6)
C = np.arange(4*5*6).reshape(4, 5, 6)
print("\n张量 C 的形状：", C.shape)

# 1. 张量 A 与张量 B 的张量积
#    这里采用外积的思想，使用 np.tensordot 并指定 axes=0，
#    结果形状为 (4, 4, 5)，其中第一个 4 来自 A，后面的 (4,5) 来自 B
AB = np.tensordot(A, B, axes=0)
print("\n1. 张量 A 与张量 B 的张量积 AB 的形状：", AB.shape)
# AB[i, j, k] = A[i] * B[j, k]

# 2. 张量 B 的第二个轴与张量 C 的第二个轴的缩并
#    这里缩并 B 的轴1和 C 的轴1，使用 np.tensordot：
#    B 的形状为 (4, 5)，C 的形状为 (4, 5, 6)；
#    缩并后保留 B 的轴0 和 C 的轴0、轴2，
#    所以结果 D 的形状为 (4, 4, 6)
D = np.tensordot(B, C, axes=([1], [1]))
print("\n2. 张量 D = tensordot(B, C, axes=([1],[1])) 的形状：", D.shape)
# D[i, j, k] = sum_{r=0}^{4} B[i, r] * C[j, r, k]

# 3. 张量 D 与张量 A 的点积
#    这里将 A 与 D 的第一个轴做点积，
#    即用 np.tensordot，将 D 的轴0 与 A 的轴0缩并，结果形状为 (4,6)
DA = np.tensordot(D, A, axes=([0], [0]))
print("\n3. 张量 DA = tensordot(D, A, axes=([0],[0])) 的形状：", DA.shape)
# DA[j, k] = sum_{i=0}^{3} D[i, j, k] * A[i]

# 4. 张量 B 与其转置 B^T 的点积
#    B 的形状为 (4,5)，B^T 的形状为 (5,4)，点积结果形状为 (4,4)
BBT = np.dot(B, B.T)
print("\n4. 张量 B 与其转置 B^T 的点积 的形状：", BBT.shape)
# BBT[i,j] = sum_{k=0}^{4} B[i,k] * B[j,k]

# 为便于观察，下面打印部分结果
print("\n结果展示：")
print("AB 的部分元素：\n", AB)
print("\nD 的部分元素：\n", D)
print("\nDA 的部分元素：\n", DA)
print("\nB 与 B^T 的点积结果：\n", BBT)
