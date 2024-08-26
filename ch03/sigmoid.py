import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#ReLU
def relu(x):
    return np.maximum(0, x) # 두 입력 중 큰 값을 선택 반환

x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

# broadcast
t = np.array([1.0, 2.0, 3.0])
print(1.0 + t)
print(1.0 / t)

x2 = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x2)
plt.plot(x2, y)
plt.ylim(-0.1, 1.1)
plt.show()

y2 = relu(x2)
plt.plot(x2, y2)
plt.ylim(-0.5, 6)
plt.show()