# 2.4 反向传播

* 反向传播: 训练模型参数，在所有参数上用梯度下降，使`NN`模型在训练数据上的损失函数最小。

* 损失函数`loss`: 计算得到的预测值`y` 与已知答案`y_`的差距。

损失函数的计算有很多方法，均方误差`MSE` 是比较常用的方法之一。

* 均方误差`MSE`: 求前向传播计算结果与已知答案之差的平方再求平均。

$$MSE(y\_,y)=\frac{\sum_{i=1}^n(y-y\_)^2}{n}$$

用tensorflow函数表示为:

```python
loss_mse = tf.reduce_mean(tf.square(y_-y))
```

* 反向传播训练方法:以减小`loss`值为优化目标，有梯度下降、`momentum`优化 器、`adam`优化器等优化方法。

这三种优化方法用tensorflow的函数可以表示为:

```python
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) 
train_step=tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)
train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss)
```

这三种优化发放区别如下：

(1)`tf.train.GradientDescentOptimizer()`使用随机梯度下降算法，使参数沿着 梯度的反方向，即总损失减小的方向移动，实现更新参数。

<img src="http://ovhbzkbox.bkt.clouddn.com/2018-08-07-15335987490795.jpg" width="250">

参数更新公式是：

$$\theta_{n+1}=\theta_n-\alpha\frac{\partial{J(\theta_n)}}{\partial\theta_n}$$

其中，$$J(\theta)$$为损失函数，$$\theta$$为参数，$$\alpha$$为学习率。

(2) `tf.train.MomentumOptimizer()`在更新参数时，利用了超参数，参数更新公式是

$$d_i=\beta{d_{i-1}}+g(\theta_{i-1})$$

$$\theta_i=\theta_{i-1}-\alpha{d_i}$$

其中，$$\alpha$$为学习率，超参数为$$\beta$$，$$g(\theta_{i-1})$$为损失函数的梯度。

(3) `tf.train.AdamOptimizer()`是利用自适应学习率的优化算法，`Adam`算法和随机梯度下降算法不同。随机梯度下降算法保持单一的学习率更新所有的参数，学习率在训练过程中并不会改变。而`Adam`算法通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率。

* 学习率:决定每次参数更新的幅度。

优化器中都需要一个叫做学习率的参数，使用时，如果学习率选择过大会出现震荡不收敛的情况，如果学习率选择过小，会出现收敛速度慢的情况。我们可以选个比较小的值填入，比如0.01、0.001。

![sgd](http://ovhbzkbox.bkt.clouddn.com/2018-07-26-sgd.gif)
![sgd_bad](http://ovhbzkbox.bkt.clouddn.com/2018-07-26-sgd_bad.gif)

