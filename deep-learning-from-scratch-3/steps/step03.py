# step03 함수 연결

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function():

    # Variable에서 데이터 찾기 / 계산 결과를 Variable에 포장하기
    def __call__(self, input): 
        x = input.data
        y = self.forward(x) # 구체적인 계산은 forward 메서드에서 한다.
        output = Variable(y)
        return output

    def forward(self, x):
        # 이 메서드는 상속하여 구현해야 한다는 사실 알림
        return NotImplementedError()

# Function 클래스를 상속받은 다양한 함수 구현
class Square(Function):
    def forward(self, x):
        return x**2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
# 함수 연결(합성함수, composite function)
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)