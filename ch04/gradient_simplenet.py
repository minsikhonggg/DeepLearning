import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화 2X3
    
    # 예측
    def predict(self, x):
        return np.dot(x, self.W)

    # 손실 함수의 값
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t) # 정답과 오차

        return loss
    
net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p)) # 최댓값 인덱스

t = np.array([0,0,1]) # 정답 레이블이라면,, (인터프리터로 확인)
net.loss(x, t)

def f(W):
    return net.loss(x, t)
# f = lambda w: net.loss(x,t)

dW = numerical_gradient(f, net.W) # 다차원 배열 처리 가능하도록 함수 수정.
print(dW)


