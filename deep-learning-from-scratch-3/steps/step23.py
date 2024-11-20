# step23 패키지로 정리
'''
- 모듈
    - 파이썬 파일
    - 다른 파이썬 프로그램에서 Import 하여 사용하는 것을 가정하고 만들어진 파일
- 패키지
    - 여러 모듈을 묶은 것
    - 디렉터리를 만들고 그 안에 모듈(파이썬 파일)을 추가해서 만듬
- 라이브러리
    - 여러 패키지를 묶은 것
    - 하나 이상의 디렉터리로 구성, 때로는 패키지를 가리켜 '라이브러리'라고 부름

- dezero 디렉터리에 모듈 추가 -> 패키지 (프레임 워크)
    - dezoro
        - __init__.py
        - core_simple.py
        - ...
        - utils.py
    - steps
        - step01.py
        - ...
        - step60.py
'''


if '__file__' in globals(): # __file__ 전역 변수 정의 되어있는지 확인
    import os, sys
    # 현재파일이 위치한 디렉터리의 부모 디렉터리(..)를 모듈 검색 경로에 추가
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
# from dezero.core_simple import Variable
from dezero import Variable # __init__에서 임포트, core_simple 생략 가능

x = Variable(np.array(1.0))
y = (x + 3) ** 2 
y.backward()

print(y) # variable(16.0)
print(x.grad) # 8.0