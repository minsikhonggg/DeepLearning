# step10 테스트

import unittest
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

# 수치미분
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

# 테스트
class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0) # 손으로 계산 하드코딩
        self.assertEqual(y.data, expected)
    
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0) # 손으로 계산 하드코딩
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1)) # 무작위 값 생성
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x) # (수치)미분 값 자동 계산
        flg = np.allclose(x.grad, num_grad) # 역전파와 수치미분 결과 비교
        self.assertTrue(flg)

# terminal
# -> python -m unittest step10.py
# -> python step10.py
unittest.main()
