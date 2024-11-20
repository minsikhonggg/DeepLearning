'''
- Class
    - Config
    - Variable
    - Function
    - Add(Function)
    - Mul(Function)
    - Neg(Function)
    - Sub(Function)
    - Div(Function)
    - Pow(Function)

- Python Function
    - using_config
    - no_grad
    - as_array
    - as_variable
    - add
    - mul
    - neg
    - sub
    - rsub
    - div
    - rdiv
    - pow
'''

import numpy as np
import weakref
import contextlib


''' ------------------------------ Class ------------------------------ '''
# 역전파 활성 여부
class Config:
    enable_backprop = True

class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
            
        self.data = data
        self.name = name # 변수 이름 지정(구분)
        self.grad = None # 미분 값 저장
        self.creator = None
        self.generation = 0 # 세대 수를 기록하는 변수

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9) # 줄바꿈 후, 여러 줄 출력 가지런히 정렬
        return 'variable(' + p + ')'
    
    # # 연산자 오버로드 -> Variable.__mul__ = mul
    # def __mul__(self, other):
    #     return mul(self, other)
    
    # @ -> 메서드를 인스턴스 변수처럼 사용 가능 
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # 세대를 기록한다(부모 세대 + 1).
    
    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            # 자동으로 미분 값 생성
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set() # 집합, 같은 함수 중복 추가 방지

        # 중첩함수 구현
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation) # 세대 수 오름차순 정렬
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop() # 함수를 가져온다.

            gys = [output().grad for output in f.outputs] # 미분 값들을 리스트에 담는다 / 약한 참조 데이터 접근()
            gxs = f.backward(*gys) # 언팩
            if not isinstance(gxs, tuple): # 튜플 처리
                gxs = (gxs,)

            # 역전파로 전달되는 미분값을 Variable의 인스턴스 변수 grad에 저장
            for x, gx in zip(f.inputs, gxs): # f.inputs[i]의 미분값은 gxs[i]에 대응

                # 미분 값 덮어씌어지기 방지
                if x.grad is None:
                    x.grad = gx
                else:
                    # x.grad += gx # 덮어 씌어진다:in-place operation / y.grad, x.grad가 같은 값 참조하는 문제
                    x.grad = x.grad + gx # 다른 위치에 복사.
                
                if x.creator is not None:
                    # funcs.append(x.creator) # 하나 앞의 함수를 리스트에 추가한다.
                    add_func(x.creator) # 중첩함수
            
            # 중간 변수의 미분 값을 모두 None으로 재설정
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # y는 약한 참조   

class Function():
    __array_priority__ = 200 # '연산자 우선순위', (우항일 경우에도)Variable 인스턴스의 연산자 메서드 우선 호출

    # Variable에서 데이터 찾기 / 계산 결과를 Variable에 포장하기
    def __call__(self, *inputs): # 별표, 임의 개수의 인수를 건네 함수 호출 가능
        inputs = [as_variable(x) for x in inputs] # inputs에 담긴 원소 x를 Variable인스턴스로 변환

        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 구체적인 계산은 forward 메서드에서 한다. / 별표를 붙여 언팩
        if not isinstance(ys, tuple): # 튜플이 아닌 경우 추가 지원
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] # 입력이 스칼라인 경우 ndarray 인스턴스로 변환

        # 노드 순서 / 계산 연결 코드는 역전파에 사용됨
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs]) # 세대 설정 / 가장 큰 generation 선택

            for output in outputs:
                output.set_creator(self) # 연결 설정 / 출력 변수에 창조자를 설정한다.

            self.inputs = inputs # 입력 변수를 기억(보관)한다.
            self.outputs = [weakref.ref(output) for output in outputs] # 출력도 저장한다.

        # 리스트 원소가 하나라면, 첫 번째 원소를 반환한다.
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        # 이 메서드는 상속하여 구현해야 한다는 사실 알림
        return NotImplementedError()
    
    def backward(self, gys):
        return NotImplementedError()


# # Function 클래스를 상속받은 다양한 함수 구현
# class Square(Function):
#     def forward(self, x):
#         return x**2
    
#     def backward(self, gy):
#         x = self.inputs[0].data # 수정전 x = self.input.data
#         gx = 2*x * gy # (미분 값 x 전달 받은 미분 값) -> 입력에 가까운 함수로 전달 
#         return gx

# class Exp(Function):
#     def forward(self, x):
#         return np.exp(x)
    
#     def backward(self, gy):
#         x = self.input.data
#         gx = np.exp(x) * gy # (미분 값 x 전달 받은 미분 값) -> 입력에 가까운 함수로 전달 
#         return gx

# 연산자 오버로드
class Add(Function):
    # def forward(self, xs):
    #     x0, x1 = xs
    #     y = x0 + x1
    #     return (y,) # 튜플로 반환

    # 두 번째 개선
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0
    
class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy
    
class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1
    
class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x**(c-1) * gy
        return gx


''' ------------------------------ Python Function ------------------------------ '''
# with를 사용한 역전파 활성 / 비활성 
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value) # with 블록 들어갈때, 지정한 value값으로 설정
    try:
        yield
    finally:
        setattr(Config, name, old_value) # with 블록 나갈때, 지정한 old_value값으로 설정

# 기울기가 필요없을때 호출하는 편의 함수, with와 함께 / 순전파 계산만 진행
def no_grad():
    return using_config('enable_backprop', False)

# 입력이 스칼라인 경우 -> ndarray 인스턴스로 변환
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# ndarray 인스턴스가 주어지면 -> Variable 인스턴스로 변환
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

# # 클래스를 '파이썬 함수'로 사용
# def square(x):
#     # f = Square()
#     # return f(x)
#     return Square()(x)

# def exp(x):
#     # f = Exp()
#     # return f(x)
#     return Exp()(x)

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0) # x0 와 x1의 순서를 바꾼다

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0) # x0 와 x1의 순서를 바꾼다

def pow(x, c):
    return Pow(c)(x)

# # 수치미분
# def numerical_diff(f, x, eps=1e-4):
#     x0 = Variable(x.data - eps)
#     x1 = Variable(x.data + eps)
#     y0 = f(x0)
#     y1 = f(x1)
#     return (y1.data - y0.data) / (2*eps)

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add # 좌/우 항을 바꿔도 결과는 똑같다.
    Variable.__mul__ = mul
    Variable.__rmul__ = mul # 좌/우 항을 바꿔도 결과는 똑같다.
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub # 좌/우항이 바뀌면 값이 바뀐다
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv # 좌/우항이 바뀌면 값이 바뀐다
    Variable.__pow__ = pow

