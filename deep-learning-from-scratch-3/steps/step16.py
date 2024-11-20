# step16 복잡한 계산 그래프(구현 편)
'''
- 복잡한 연결의 역전파 구현
    - 분기, 합류
    - 역전파의 순서
- 우선순위 부여
    - 함수의 세대 generation

- 중첩함수
    - 감싸는 메서드(backward 메서드) 안에서만 이용한다.
    - 감싸는 메서드(backward 메서드)에 정의된 변수(funcs과 seen_set)를 사용해야 한다.
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
        self.generation = 0 # 세대 수를 기록하는 변수
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # 세대를 기록한다(부모 세대 + 1).
    
    def cleargrad(self):
        self.grad = None

    def backward(self):
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

            gys = [output.grad for output in f.outputs] # 미분 값들을 리스트에 담는다
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

class Function():
    # Variable에서 데이터 찾기 / 계산 결과를 Variable에 포장하기
    def __call__(self, *inputs): # 별표, 임의 개수의 인수를 건네 함수 호출 가능
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 구체적인 계산은 forward 메서드에서 한다. / 별표를 붙여 언팩
        if not isinstance(ys, tuple): # 튜플이 아닌 경우 추가 지원
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] # 입력이 스칼라인 경우 ndarray 인스턴스로 변환

        self.generation = max([x.generation for x in inputs]) # 가장 큰 generation 선택

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
        x = self.inputs[0].data # 수정전 x = self.input.data
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
    
    def backward(self, gy):
        return gy, gy


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


# --------------실행 부분--------------

x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data) # 32.0
print(x.grad) # 64.0