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