if '__file__' in globals(): # __file__ 전역 변수 정의 되어있는지 확인
    import os, sys
    # 현재파일이 위치한 디렉터리의 부모 디렉터리(..)를 모듈 검색 경로에 추가
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# step24. 복잡한 함수의 미분
import numpy as np
from dezero import Variable

def sphere(x, y):
    z = x**2 + y**2
    return z

def matyas(x, y):
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    # 연산자를 사용 못한다면 아래와 같이.. 복잡
    # z = sub(mul(0.26, add(pow(x,2), pow(y,2))), mul(0.48, mul(x,y)))
    return z

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

# --------------------------실행 부분-------------------------- #

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x, y)
z.backward()
print(x.grad, y.grad) # 2.0 2.0

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = matyas(x, y)
z.backward()
print(x.grad, y.grad) # 0.040000000000000036 0.040000000000000036

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x, y)
z.backward()
print(x.grad, y.grad) # -5376.0 8064.0