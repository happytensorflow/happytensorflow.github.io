# 2.5 搭建神经网络的步骤

我们最后梳理出神经网络搭建的八股，神经网络的搭建课分四步完成:准备工作、 前向传播、反向传播和循环迭代。

* 导入模块，生成模拟数据集

```python
import 
常量定义 
生成数据集
```

* 前向传播: 定义输入、参数和输出

```
x=   y_=w1=  w2=a=   y=
```

* 反向传播: 定义损失函数、反向传播方法

```
loss=train_step=
```

* 生成会话，训练`STEPS`轮

```
with tf.session() as sess:  init_op=tf. global_variables_initializer() 
  sess_run(init_op)
  STEPS=3000  for i in range(STEPS):    start=    end=    sess.run(train_step, feed_dict:)
```

举例

随机产生32组生产出的零件的体积和重量，训练3000轮，每500轮输出一次损 失函数。下面我们通过源代码进一步理解神经网络的实现过程: 

(0) 导入模块，生成模拟数据集;

```python
#encoding:utf-8
#0 导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed = 23455

# 基于seed产生随机数
rng = np.random.RandomState(seed)

# 随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入数据集
X = rng.rand(32,2)

# 从X这个32行2列的矩阵中 取出一行 判断如果和小于1 给Y赋值1 如果不小于1 给Y赋值0
# 作为输入数据集的标签(正确答案)
Y = [[int(x0 + x1 < 1)] for (x0,x1) in X]
print("X:\n",X)
print("Y:\n",Y)
```

(1)定义神经网络的输入、参数和输出，定义前向传播过程;

```python
#1 定义神经网络的输入、参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
```

(2)定义损失函数及反向传播方法

```python
#2 定义损失函数及方向传播方法
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.01,0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
```

(3)生成会话，训练`STEPS`轮

```python
#3 生成会话，训练STEP轮
with tf.Session() as sess:
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  # 输出目前(未经训练)的参数取值
  print("w1:\n", sess.run(w1))
  print("w2:\n", sess.run(w2))
  
  # 训练模型
  STEPS = 3000
  for i in range(STEPS):
    start = (i*BATCH_SIZE) % 32
    end = start + BATCH_SIZE
    sess.run(train_step, feed_dict={x:X[start:end], y_: Y[start:end]})
    if i % 500 == 0:
      total_loss = sess.run(loss, feed_dict={x:X,y_:Y})
      print("After %d training step(s), loss on all data is %g" % (i, total_loss))
    # 输出训练后的参数取值
    print("\n")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
```

由神经网络的实现结果，我们可以看出，总共训练3000轮，每轮从`X`的数据集和`Y`的标签中抽取相对应的从`start`开始到`end`结束个特征值和标签，喂入神经网络，用 `sess.run`求出`loss`，每500轮打印一次 `loss`值。经过3000轮后，我们打印出最终训练好的参数`w1`、`w2`。

```
After 0 training step(s), loss on all data is 5.13118
After 500 training step(s), loss on all data is 0.429111
After 1000 training step(s), loss on all data is 0.409789
After 1500 training steps(s), loss on all data is 0.399923
After 1500 training steps(s), loss on all data is 0.394146
After 1500 training steps(s), loss on all data is 0.390597

w1:
[[-0.70006633 0.9136318  0.08953571
 [-2.3402493 -0.14641267 0.58823055]] 
w2:
[[-0.06024267]
 [ 0.91956168]
 [-0.0682071 ]]
```

这样四步就可以实现神经网络的搭建了。


