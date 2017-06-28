#17/06/22課題 sin(x)の回帰
#入力30個 隠れ40個 出力30個
#隠れ層の活性化関数はシグモイド、出力層に恒等写像を使用
#50000回のループ処理

#参考
#http://tadaoyamaoka.hatenablog.com/entry/2016/04/10/120305
#↑入力が行ベクトルなので注意

#後からfor文を書き足すときにインデントを一括して入れる方法が知りたい

import numpy as np
import matplotlib.pyplot as plt
x = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1], [1.1], [1.2], [1.3], [1.4], [1.5], [1.6], [1.7], [1.8], [1.9], [2.0], [2.1], [2.2], [2.3], [2.4], [2.5], [2.6], [2.7], [2.8], [2.9], [3.0]])
w1 = np.random.randn(40,30)
w2 = np.random.randn(30,40)
def sigmoid(u):
    return 1/(1+np.exp(-u))

#1周期分は欲しいので3倍して入力
x2 = 3*x
before_lerning = w2.dot(sigmoid(w1.dot(x2)))
d = np.sin(x2)

i = 1
for i in range(1, 50000):
    u2 = w1.dot(x2)
    z2 = sigmoid(u2)
    z3 = w2.dot(z2)
    grad_w2 = (z3-d).dot(z2.T)
    w2 -= 0.00001*grad_w2
    delta1 = w2.T.dot(z3-d)
    sigmoid_dash = z2*(1-z2)
    grad_w1 = delta1*sigmoid_dash
    grad_w11 = grad_w1.dot(x2.T)
    w1 -= 0.00001*grad_w11
    print (i)

plt.plot(x2, z3)
plt.plot(x2, before_lerning)
plt.show()
