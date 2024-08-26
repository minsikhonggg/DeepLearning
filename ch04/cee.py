# cee.py

import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 정답 2


# 예1: 2일 확률이 가장 높다고 추정(0.6)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] 
print(cross_entropy_error(np.array(y), np.array(t))) 
0.510825457099338

# 예2: 7일 확률이 가장 높다(0.6)
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] 
print(cross_entropy_error(np.array(y), np.array(t))) 
2.302584092994546

