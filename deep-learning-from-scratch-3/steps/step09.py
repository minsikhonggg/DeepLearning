# step09 함수를 더 편리하게
'''
- 세 가지 개선
    1. 파이썬 함수로 이용하기
    2. backward 메서드 간소화
    3. ndarray만 취급하기
'''

import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
            
        self.data = data
        self.grad = None # 미분 값 저장
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            # 자동으로 미분 값 생성
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 함수를 가져온다.
            x, y = f.input, f.output # 함수의 입력과 출력을 가져온다.
            x.grad = f.backward(y.grad) # backward 메서드를 호출한다.
            
            if x.creator is not None:
                funcs.append(x.creator) # 하나 앞의 함수를 리스트에 추가한다.

class Function():

    # Variable에서 데이터 찾기 / 계산 결과를 Variable에 포장하기
    def __call__(self, input): 
        x = input.data
        y = self.forward(x) # 구체적인 계산은 forward 메서드에서 한다.
        output = Variable(as_array(y)) # 입력이 스칼라인 경우 ndarray 인스턴스로 변환
        output.set_creator(self) # 출력 변수에 창조자를 설정한다.
        self.input = input # 입력 변수를 기억(보관)한다.
        self.output = output # 출력도 저장한다.
        return output

    def forward(self, x):
        # 이 메서드는 상속하여 구현해야 한다는 사실 알림
        return NotImplementedError()
    
    def backward(self, dy):
        return NotImplementedError()

# Function 클래스를 상속받은 다양한 함수 구현
class Square(Function):
    def forward(self, x):
        return x**2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2*x * gy # (미분 값 x 전달 받은 미분 값) -> 입력에 가까운 함수로 전달 
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy # (미분 값 x 전달 받은 미분 값) -> 입력에 가까운 함수로 전달 
        return gx

def square(x):
    # f = Square()
    # return f(x)
    return Square()(x)

def exp(x):
    # f = Exp()
    # return f(x)
    return Exp()(x)

def as_array(x):
    # 입력이 스칼라인 경우 -> ndarray 인스턴스로 변환
    if np.isscalar(x):
        return np.array(x)
    return x



# 순전파
x = Variable(np.array(0.5))
a = square(x) # 파이썬 함수로 이용 가능
b = exp(a) 
y = square(b)

# 역전파 자동 실행
y.grad = np.array(1.0)
y.backward()
print(x.grad)

# 순전파
x2 = Variable(np.array(0.5))
y2 = square(exp(square(x2))) # 연속하여 적용 가능

# 역전파 자동 실행
y2.grad = np.array(1.0)
y2.backward()
print(x2.grad)

# backward 호출만으로 미분 값 계산(최초 역전파 값 설정X)
x3 = Variable(np.array(0.5))
y3 = square(exp(square(x3)))
y3.backward()
print(x3.grad)

x = Variable(np.array(1.0)) # OK
x = Variable(None) # OK
# x = Variable(1.0) # Error