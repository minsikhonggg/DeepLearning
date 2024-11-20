# step12 가변 길이 인수(개선 편)
'''
개선1. 함수를 사용하기 쉽게
개선2. 함수를 구현하기 쉽도록
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
    def __call__(self, *inputs): # 별표, 임의 개수의 인수를 건네 함수 호출 가능
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 구체적인 계산은 forward 메서드에서 한다. / 별표를 붙여 언팩
        if not isinstance(ys, tuple): # 튜플이 아닌 경우 추가 지원
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] # 입력이 스칼라인 경우 ndarray 인스턴스로 변환

        for output in outputs:
            output.set_creator(self) # 출력 변수에 창조자를 설정한다.

        self.inputs = inputs # 입력 변수를 기억(보관)한다.
        self.outputs = outputs # 출력도 저장한다.

        # 리스트 원소가 하나라면, 첫 번째 원소를 반환한다.
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        # 이 메서드는 상속하여 구현해야 한다는 사실 알림
        return NotImplementedError()
    
    def backward(self, gys):
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

class Add(Function):
    # def forward(self, xs):
    #     x0, x1 = xs
    #     y = x0 + x1
    #     return (y,) # 튜플로 반환

    # 두 번째 개선
    def forward(self, x0, x1):
        y = x0 + x1
        return y


# 클래스를 '파이썬 함수'로 사용
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

def add(x0, x1):
    return Add()(x0, x1)

# 수치미분
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)


# xs = [Variable(np.array(2)), Variable(np.array(3))] # 리스트로 준비
# f = Add()
# ys = f(xs) # ys 튜플
# y = ys[0]
# print(y.data)

# 첫 번째 개선
x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
f = Add()
y = f(x0, x1)
print(y.data)

# 두 번째 개선
x0 = Variable(np.array(5))
x1 = Variable(np.array(10))
y = add(x0, x1) # Add 클래스 생성 과정이 감춰짐
print(y.data)