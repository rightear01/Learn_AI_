# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

mid_scores = [10, 20, 30]
final_scores = [70, 80, 90]
total = (mid_scores + final_scores)
print(total)

# * 이것은 우리가 원하는 연산이 아니다.

import numpy as np

mid_scores = np.array([10, 20, 30])
final_scores = np.array([70, 80, 90])

total = (mid_scores + final_scores)
print(total)

import numpy as np
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
z = x + y

print(z)

# # *BMI 지수 계산하기

# +
heights = [1.83, 1.76, 1.69, 1.86, 1.77, 1.73]
weights = [86, 74, 59, 95, 80, 68]

np_heights = np.array(heights)
np_weights = np.array(weights)

bmi = np_weights/(np_heights**2)
print(bmi)
# -

import numpy as np
y = [[1,2,3],[4,5,6],[7,8,9]]
print(y, "\n")
ny = np.array(y)
print(ny)

print(ny[0][2])

print(np.arange(5))

print(type(np.arange(5)))

print(np.linspace(0,10,100))

y = np.arange(12)
print(y)

y = y.reshape(3,4)
print(y)

y = y.reshape(6,-1)
print(y)

# * reshape() -> 데이터의 개수는 유지한 채로 배열의 차원과 형태를 변경한다.
# y = y.reshape(6,-1) 인수로 -1을 전달하면 데이터의 개수에 맞춰서 자동으로 배열의 형태가 결정된다.

np.random.seed(100)
np.random.rand(5)
np.random.rand(10)

# np.random.seed() -> np.random의 모듈 추출결과를 고정한다.(출력값들이 고정된다.)

import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.rand(5,3)
plt.figure(figsize = (5, 3)) # figsize 는 그래프의 크기이다. 가로 세로 figure는 그림 전체
print(x1)
plt.hist(x1, bins = 15, alpha = 1.0) # bin은 박스의 개수, alpha는 투명도이다.
plt.show()

# 난수로 이루어진 2차원 배열(5 x 3)을 얻는 방법이다.

a = 10; b = 20
(b-a)*np.random.rand(5)+a

# * 위 방법은 범위에 있는 난수를 생성하는 방법이다. 위는 10 -20 사이에 있는 난수 5개를 생성하는 문장이다.

np.random.randint(1, 7, size = 10)

# * 정수형 난수를 생성시키는 방법은 randint()를 사용하면 된다.

np.random.randint(1, 11, size = (4,7))

# * size 매개변수를 이용하여 행렬의 사이즈를 조절할 수 있다. 위는 4행 7열이다.

# # 위의 난수들은 균일한 확률의 분포에서 만들어짐

# # how to make normal distribution?

# * 넘파이 라이브러리에는 randn()함수가 있다.

np.random.randn(5)

np.random.randn(5,4)

# # 정규분포에서 생성함 행렬의 크기 또한 지정가능!
# * 위의 정규분포는 평균이 0이고 표준편차가 1이다.

m = 10; sigma = 2 # 평균 10 표준편차 2인 정규분포에서 난수를 생성
(m + sigma)*np.random.rand(5)

# # 정규분포 그래프 그리기

import numpy as np
import matplotlib.pyplot as plt

m = 10; sigma = 2
x1 = np.random.randn(1000)
x2 = m + sigma*np.random.randn(1000)
plt.figure(figsize = (5, 3)) # figsize 는 그래프의 크기이다. 가로 세로 figure는 그림 전체
plt.hist(x1, bins = 20, alpha = 1.0) # bin은 박스의 개수, alpha는 투명도이다.
plt.hist(x2, bins = 20, alpha = 0.4) # 20개의 상자를 이용하여 히스토그램 계산
plt.show()

# # 밑에서부터는 선형회귀 관련임.

# +
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.0, 1.0, 2.0])
y = np.array([3.0, 3.5, 5.5])

w = 0         # 기울기 * 가중치
b = 0         # y절편 * 바이어스

rating = 0.01     # 학습률
epochs = 1000     # 반복 횟수
n = float(len(x)) # 입력데이터의 개수

# 경사하강법
for i in range(epochs):
    y_pred = w * x + b                  # 예측값 f(x) = Wx + b -> 선형 회귀의 문제는 직선의 방정식
    dw = (2/n) * sum(x * (y_pred - y)) # 어느 시점의 가중치 변화량
    w = w - rating * dw                # 가중치 갱신
    db = (2/n) * sum(y_pred - y)      # 어느 시점의 바이어스 변화량
    b = b - rating * db               # 바이어스 갱신
    
# 기울기와 절편을 출력한다.
print(w, b)

# 예측값을 만든다.
y_pred = w * x + b

# 입력 데이터를 그래프 상에 찍는다.
plt.figure(figsize = (5, 3)) # figsize 는 그래프의 크기이다. 가로 세로 figure는 그림 전체
plt.scatter(x, y)
print(x)
print(y)

#예측값은 선그래프로 그린다.
plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color = 'red')
plt.show()

# +
import matplotlib.pylab as plt
import numpy as np
from sklearn import linear_model

# 선형 회귀 모델을 생성한다
reg = linear_model.LinearRegression()

# 데이터는 파이썬의 리스트로 만들어도 되고 아니면 넘파이의 배열로 만들어도 된다.
x = np.array([[0], [1], [2]]) # 2차원으로 만들어야 한다.
y = np.array([3, 3.5, 5.5])   # y = x + 3 & 목적값은 1차원이어도 된다.

# 학습을 시킨다.
reg.fit(x, y)
# -

reg.coef_ # 직선의 기울기

reg.intercept_

reg.score(x,y)

reg.predict([[5]]) # 예측값은 9

# +
# 학습 데이터와 y 값을 산포도로 그린다.
plt.scatter(x, y, color = 'black')

#학습 데이터를 입력으로 하여 예측값을 계산한다.
y_pred = reg.predict(x)

# 학습 데이터와 예측값으로 선 그래프를 그린다.
# 계산된 기울기와 y절편을 가지는 직선이 그려진다.
plt.plot(x, y_pred, color = 'blue', linewidth = 3)
plt.show()

# +
import matplotlib.pylab as plt
from sklearn import linear_model

reg = linear_model.LinearRegression()

x = [[174], [152], [138], [128], [186]]
y = [71, 55, 46, 38, 88]
reg.fit(x, y)

print(reg.predict([[165]]))

# 학습 데이터와 y 값을 산포도로 그린다.
plt.scatter(x, y, color = 'black')

# 학습 데이터를 입력으로 하여 예측값을 계산한다.
y_pred = reg.predict(x)

# 학습 데이터와 예측값으로 선 그래프로 그린다.
# 계산된 기울기와  y 절편을 가지는 직선이 그려진다
plt.plot(x, y_pred, color = 'blue', linewidth = 3)
plt.show()
# -

# # 위 문제는 인간의 키와 몸무게를 예상하여 165일때 예측값을 얻어보는 문제이다.

# # * 당뇨병 예제

# +
import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets # sklearn에서 기본적으로 제공하는 당뇨병 환자들 데이터 셋

# 당뇨병 데이터 세트를 적재한다.
datasets = datasets.load_diabetes()
datasets.data.shape
# -

datasets.feature_names

# +
# 학습 데이터와 테스트 데이터를 분리한다.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(datasets.data, datasets.target, test_size=0.2, random_state = 0)

#선형 회귀 모델로 학습을 수행한다.
model = LinearRegression()
model.fit(X_train, y_train)

#학습이 끝난 후에 선형 회귀 모델을 사용하여 예측을 해보자. 이때 아껴두었던 테스트 데이터를 사용한다.

#테스트 데이터로 예측해보자
y_pred = model.predict(X_test)

#결과를 그래프로 그려서 비교해보자

#실제 데이터와 예측 데이터를 비교해보자
plt.figure(figsize = (20, 20))
plt.plot(y_test, y_pred, '.')

# y_test와 y_pred가 일치하면 가장 좋은 것이다. 하지만 완벽하게 일치하지는 않을 것이다, 이것을 알아보기 위하여 
# 기울기가 1인 직선을 그려보자. 모든 점들이 직선 위에 있으면 가장 좋다. (y = x)

# 직선을 그리기 위하여 완벽한 선형 데이터를 생성한다.
print(x)
y = x # y = x 그래프 -- 기울기가 1인 그래프를 그리기 위함.
plt.plot(x, y, "k--")
plt.show()
# -


