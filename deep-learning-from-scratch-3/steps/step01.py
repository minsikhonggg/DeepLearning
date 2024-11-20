# step01. 상자로서의 변수
'''
- 상자와 데이터는 별개다.
- 상자에는 데이터가 들어간다(대입 혹은 할당한다).
- 상자 속을 들여다보면 데이터를 알 수 있다(참조한다).
'''

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

data = np.array(1.0)
x = Variable(data)
print(x.data)

# 새로운 데이터 대입
x.data = np.array(2.0)
print(x.data)

x.data = np.array([1,2,3])
print(x.data)

# ndim(number of dimensions), 다차원 배열의 차원 수
print(x.data.ndim) 