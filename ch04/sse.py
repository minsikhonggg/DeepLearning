# sse.py

import numpy as np

def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 정답 2


# 예1: 2일 확률이 가장 높다고 추정(0.6)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] 
print(sum_squares_error(np.array(y), np.array(t))) 
# 0.09750000000000003

# 예2: 7일 확률이 가장 높다(0.6)
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] 
print(sum_squares_error(np.array(y), np.array(t))) 
# 0.5975

