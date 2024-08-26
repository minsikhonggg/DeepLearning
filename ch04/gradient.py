# gradient.py

import numpy as np

def function_2(x):
    return x[0]**2 + x[1]**2
    # 또는 return np.sum(x**2)

# 기울기 구하기
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
    
    return grad

# 경사 하강법
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x) # 기울기 구하기
        x -= lr * grad # 갱신: 학습률x기울기
    
    return x

# 기울기 구하기
print(numerical_gradient(function_2, np.array([3.0,4.0]))) # (3,4) 기울기, 7.999999999999119
print(numerical_gradient(function_2, np.array([0.0,2.0]))) # (0,2) 기울기, 4.000000000004
print(numerical_gradient(function_2, np.array([3.0,0.0]))) # (3,0) 기울기, 0.0

# 경사 하강법으로 최솟값 구하기
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
# [-7.00000000e+00  8.12235612e-10] -> 거의 0,0에 가까운 결과

# 경사 하강법으로 최솟값 구하기(학습률이 너무 크다면? lr=10.0)
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))
# [-1.29504382e+12 -1.29504382e+12] -> 발산

# 경사 하강법으로 최솟값 구하기(학습률이 너무 작다면? lr=1e-10)
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))
# [-3.00000008  3.99999992] -> 거의 갱신X