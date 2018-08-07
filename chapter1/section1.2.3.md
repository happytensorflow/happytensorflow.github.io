# 2.3 前向传播

* 前向传播就是搭建模型的计算过程，让模型具有推理能力，可以针对一组输入 给出相应的输出。

举例: 假如生产一批零件，体积为$$x_1$$，重量为$$x_2$$，体积和重量就是我们选择的特征， 把它们喂入神经网络，当体积和重量这组数据走过神经网络后会得到一个输出。
假如输入的特征值是: 体积`0.7`重量`0.5`

![](http://ovhbzkbox.bkt.clouddn.com/2018-07-17-15318051293124.jpg)

由搭建的神经网络可得，隐藏层节点 `a11=x1* w11+x2*w21=0.14+0.15=0.29`，同理算得节点 `a12=0.32`，`a13=0.38`，最终计算得到输出层`Y=-0.015`，这便实现了 前向传播过程。

* 推导

#### 第一层

* `X`是输入为`1X2`的矩阵:

>用`x`表示输入，是一个1行2列矩阵，表示一次输入一组特征，这组特征包含了体积和重量两个元素。
 

* $$W_{pre-node,post-node}^{(layer)}$$为待优化的参数

>对于第一层的w 前面有两个节点，后面有三个节点 w应该是个两行三列矩阵，我们这样表示:

$$
w^{(1)}=
\left[ \begin{array}{ccc}
w_{1,1}^{(1)} & w_{1,2}^{(1)} & w_{1,3}^{(1)}\\
w_{2,1}^{(1)} & w_{2,2}^{(1)} & w_{2,3}^{(1)}
\end{array}\right ]
$$

* 神经网络共有几层(或当前是第几层网络)都是指的计算层，输入不是计算层， 所以`a`为第一层网络，`a`是一个一行三列矩阵。
我们这样表示:

$$a^{(1)}=[a_{11},a_{12},a_{13}]=XW^{(1)}$$


#### 第二层

* 参数要满足前面三个节点，后面一个节点，所以$$W^{(2)}$$是三行一列矩阵。我们这样表示：

$$
W^{(2)}=\left[ \begin{array}{ccc}
W_{(1,1)}^{(2)}\\W_{(2,1)}^{(2)}\\W_{(3,1)}^{(2)}
\end{array}\right ]
$$

我们把每层输入乘以线上的权重`w`，这样用矩阵乘法可以计算出输出`y`了。

```python
a = tf.matmul(X, W1)
y = tf.matmul(a, W2)
```

由于需要计算结果，就要用`with`结构实现，所有变量初始化过程、计算过程都要放到 `sess.run`函数中。对于变量初始化，我们在`sess.run`中写入 `tf.global_variables_initializer` 实现对所有变量初始化，也就是赋初值。对 于计算图中的运算，我们直接把运算节点填入 `sess.run`即可，比如要计算输出`y`，直接写`sess.run(y)`即可。

在实际应用中，我们可以一次喂入一组或多组输入，让神经网络计算输出`y`，可以先用 `tf.placeholder`给输入占位。如果一次喂一组数据`shape`的第一维位置写1，第二维位置看有几个输入特征;如果一次想喂多组数据，`shape`的第一维位置可以写`None` 表示先空着，第二维位置写有几个输入特征。这样在`feed_dict`中可以喂入若干组体积重量了。

##### 前向传播过程的tensorflow描述:

变量初始化、计算图节点运算都要用会话(with 结构)实现

```python
with tf.Session() as sess:
  sess.run()
```

变量初始化:在`sess.run`函数中用 `tf.global_variables_initializer()`汇总所有待优化变量。

```python
init_op = tf.global_variables_initializer() sess.run(init_op)
```

计算图节点运算:在`sess.run`函数中写入待运算的节点`sess.run(y)`


用`tf.placeholder`占位，在`sess.run`函数中用`feed_dict`喂数据 

```python
# 喂一组数据
x = tf.placeholder(tf.float32, shape=(1, 2))sess.run(y, feed_dict={x: [[0.5,0.6]]})
```

```python
# 喂多组数据
x = tf.placeholder(tf.float32, shape=(None, 2))sess.run(y, feed_dict={x: [[0.1,0.2],[0.2,0.3],[0.3,0.4],[0.4,0.5]]})
```

举例

这是一个实现神经网络前向传播过程，网络可以自动推理出输出`y`的值。

(1)用`placeholder`实现输入定义(`sess.run`中喂入一组数据)的情况，第一组喂体积0.7、重量0.5

```python
#coding:utf-8import tensorflow as tf#定义输入和参数x=tf.placeholder(tf.float32,shape=(1,2)) 
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1)) 
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#定义前向传播过程a=tf.matmul(x,w1)y=tf.matmul(a,w2)#用会话计算结果
with tf.Session() as sess:  init_op=tf.global_variables_initializer()  sess.run(init_op)  print("y is:\n",sess.run(y,feed_dict={x:[[0.7,0.5]]}))
```

(2)用`placeholder`实现输入定义(`sess.run`中喂入多组数据)的情况第一组喂体积0.7、重量0.5，第二组喂体积 0.2、重量0.3，第三组喂体积0.3、重量 0.4，第四组喂体积0.4、重量0.5

```python
#coding:utf-8import tensorflow as tf#定义输入和参数 x=tf.placeholder(tf.float32,shape=(None,2)) w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1)) w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1)) 
#定义前向传播过程a=tf.matmul(x,w1)y=tf.matmul(a,w2)#用会话计算结果with tf.Session() as sess:  init_op=tf.global_variables_initializer()  sess.run(init_op)  print("y is:\n",sess.run(y,feed_dict={x:[[0.7,0.5], [0.2,0.3],[0.3,0.4]，[0.4,0.5]]}))
```

