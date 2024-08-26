# softmax_overflow.py

import numpy as np

a = np.array([1010, 1000, 990])
print(np.exp(a) / np.sum(np.exp(a))) # [nan nan nan]

c = np.max(a) # 최댓값 1010
print(a-c) # [  0 -10 -20]

print(np.exp(a-c) / np.sum(np.exp(a-c))) # [9.99954600e-01 4.53978686e-05 2.06106005e-09]

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

x = np.array([0.3, 2.9, 4.0])
y = softmax(x)
print(y) # 0.01821127 0.24519181 0.73659691]
print(np.sum(y)) # 1.0