# step02. 변수를 낳는 함수
'''
- Function 클래스는 Variable 인스턴스를 입력받아 Variable 인스턴스를 출력한다.
- Variable 인스턴스의 실제 데이터는 인스턴스 변수인 data에 있다.

- Function 클래스는 기반 클래스로서, 모든 함수에 공통되는 기능을 구현한다.
- 구체적인 함수는 Function 클래스를 상속한 클래스에서 구현한다.
'''

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        
# class Function():
#     def __call__(self, input):
#         x = input.data # 데이터를 꺼낸다
#         y = x**2 # 실제 계산 
#         output = Variable(y) # Variable 형태로 되돌린다.
#         return output
    
# x = Variable(np.array(10)) # Variable의 인스턴스 x
# f = Function()
# y = f(x) # Function의 인스턴스 y / 입력으로 x

# print(type(y))
# print(y.data)

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
    
# Function 클래스를 상속하여 입력값을 제곱하는 클래스 구현
class Square(Function):
    def forward(self, x):
        return x**2

x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)