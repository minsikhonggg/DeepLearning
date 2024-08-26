# numerical_differentiation.py

import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    # h = 10e-50
    # return (f(x+h) - f(x)) / h
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return x[0]**2 + x[1]**2
    # 또는 return np.sum(x**2)

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

# x=5 일 때 미분, 0.1999999999990898 / 진정한 미분 값(해석학적) 0.2
print(numerical_diff(function_1, 5))
# x=10 일 때 미분, 0.2999999999986347 / 진정한 미분 값(해석학적) 0.3
print(numerical_diff(function_1, 10))

# x0=3, x1=4일 때, x0에 대한 편미분을 구하라.
print(numerical_diff(function_tmp1, 3.0))
# 6.00000000000378

# x0=3, x1=4일 때, x1에 대한 편미분을 구하라.
print(numerical_diff(function_tmp2, 4.0))
# 7.999999999999119