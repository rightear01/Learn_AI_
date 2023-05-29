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

# +
import matplotlib.pylab as plt
import numpy as np
def calculate(X):
    global w
    global b
    activation = b # 임계값은 바이어스값의 음수를 취한 값이다.
    
    for i in range(2):
        activation += w[i] * X[i]
    if activation >= 0.0:
        return 1.0
    else:
        return 0.0

def train(X, y, lrate, n_epoch):
    global w
    global b
    for epoch in range(n_epoch): # 에폭만큼 돌리기
        sum_error = 0.0
        for row, lable in zip(X, y):
            actual = calculate(row)
            error = lable - actual  # 오류율 = 레이블에서 얼마나 떨어져 있는가
            b += lrate * error      # 바이어스 조정
            sum_error += error**2  # 오류 제곱(참고값)
            for i in range(2):
                w[i] = w[i] + lrate * error * row[i] # 가중치 조정
            print(w, b)
        print('Epoch = %d, Rating = %.3f, Error = %.3f' % (epoch, lrate, sum_error))
    return w


# +
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 1])
w = np.array([0.0, 0.0])
b = 0.0
lrate = 0.01
n_epoch = 10

w = train(X, y, lrate, n_epoch)
print(w, b)
plt.figure(figsize = (5, 3)) # figsize 는 그래프의 크기이다. 가로 세로 figure는 그림 전체
for i in range(4):
    plt.scatter(X[i][0], X[i][1])
    
x1_range = np.linspace(-0.5, 1.5, 100)
x2_range = (-b - w[0] * x1_range) / w[1]
plt.plot(x1_range, x2_range, color='red')

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('X[1]')
plt.ylabel('X[2]')
plt.title('Decision Boundary')
plt.grid(True)
plt.show()

# +
X = [[170, 80], [175, 76], [180, 70], [160, 55], [163, 42], [165, 48]]
y = [1, 1, 1, 0, 0, 0]
w = [0.0, 0.0]
b = 0.0
lrate = 0.01
n_epoch = 100

w = train(X, y, lrate, n_epoch)
print(w, b)

x1_range = np.linspace(159, 181, 100) # 키
x2_range = (-b - w[0] * x1_range) / w[1] # 몸무게
print('\n',x1_range,'\n')
print(x2_range)

plt.figure(figsize = (5, 3)) # figsize 는 그래프의 크기이다. 가로 세로 figure는 그림 전체
plt.plot(x1_range, x2_range, color='red')

for i in range(6):
    plt.scatter(X[i][0], X[i][1])

plt.xlim(159, 181)
plt.ylim(40, 90)
plt.xlabel('Height')
plt.ylabel('weight')
plt.title('Decision Boundary')
plt.grid(True)
plt.show()
# -

# ## * sklearn 라이브러리를 사용하여 퍼셉트론 구현하기

# +
from sklearn.linear_model import Perceptron

# 샘플과 레이블이다.
x = [[170, 80], [175, 76], [180, 70], [160, 55], [163, 42], [165, 48]]
y = [0, 0, 0, 1, 1, 1]

# 퍼셉트론을 생성한다. tol는 종료 조건이다 1e-3 0.001. random_state는 난수의 시드이다.
clf = Perceptron(tol=1e-3, random_state = 0)

# 학습을 수행한다.
clf.fit(x, y)

# 테스트를 수행한다.
print(clf.predict(x))
