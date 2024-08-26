import numpy as np

# 1차원 배열
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A)) # 배열의 차원수 ndim
print(A.shape) # 배열의 형상 / 튜플로 반환
print(A.shape[0])

# 2차원 배열
B = np.array([[1,2], [3,4], [5,6]])
print(B)
print(np.ndim(B))
print(B.shape)

# 행렬 곱
C = np.array([[1,2,3],[4,5,6]])
print(C.shape)
D = np.array([[1,2],[3,4],[5,6]])
print(D.shape)
print(np.dot(C,D))

# 신경망에서 행렬 곱
X = np.array([1,2])
print(X.shape)
W = np.array([[1,3,5],[2,4,6]])
print(W)
print(W.shape)
Y = np.dot(X,W)
print(Y)