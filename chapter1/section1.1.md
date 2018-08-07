# 基本概念

* 基于 Tensorflow 的 NN:用张量表示数据，用计算图搭建神经网络，用会话执 行计算图，优化线上的权重(参数)，得到模型。


* 张量:张量就是多维数组(列表)，用`阶`表示张量的维度。

| 维数 | 阶 | 名字 | 例子 |
| --- | --- | --- | --- |
| 0-D | 0 | 标量 scalar | s=123 |
| 1-D | 1 | 向量 vector | v=[1,2,3] |
| 2-D | 2 | 矩阵 matrix | m=[[1,2,3],[4,5,6],[7,8,9]] |
| n-D | n | 张量 tensor | t=[[[… （n个中括号） |

张量可以表示0阶到n阶数组(列表)

* 数据类型：`tf.float32` `tf.int32` ...

```python
import tensorflow as tf
a = tf.constant([1.0,2.0])
b = tf.constant([3.0,4.0])
result = a+b
print(result)
```

显示

```
Tensor("add:0", shape=(2,), dtype=float.32)
```

其中`add:`表示`节点名`，`0`表示`第0个输出`,`shape`表示`维度`


注意vim

```
vim ~/.vimrc写入：
set ts=4    # tab键转成4个空格
set nu      # 显示行号
```

* 计算图(Graph)：搭建神经网络的计算过程，只搭建，不运算。

<img src="http://ovhbzkbox.bkt.clouddn.com/2018-07-17-15318034642927.jpg" width="200"/>

```python
import tensorflow as tf

x = tf.constant([[1.0,2.0]])
w = tf.constant([[3.0],[4.0]])

y = tf.matmul(x,w)
print(y)
```

显示

```python
Tensor("matmul:0",shape(1,1),dtype=float32)
```

从这里我们可以看出，print 的结果显示 y 是一个张量，只搭建承载计算过程的 计算图，并没有运算，如果我们想得到运算结果就要用到“会话 Session()”了。

* 会话(Session):执行计算图中的节点运算。

我们用 with 结构实现，语法如下:

```
with tf.Session() as sess:
  print(sess.run(y))
```

举例

对于刚刚所述计算图，我们执行 Session()会话可得到矩阵相乘结果:

计算`1.0*3.0+2.0*4.0=11.0`

```python
import tensorflow as tf

x = tf.constant([[1.0,2.0]])
w = tf.constant([[3.0],[4.0]])

y = tf.matmul(x,w)
print(y)

with tf.Session() as sess:
    print(sess.run(y))
```

结果

```python
Tensor("matmul:0",shape(1,1),dtype=float32)
[[11.]]
```

我们可以看到，运行`Session()`会话前只打印出`y`是个张量的提示，运行`Session()`会话后打印出了`y`的结果`1.0*3.0 + 2.0*4.0 = 11.0`。

> 注1: 我们以后会常用到`vim`编辑器，为了使用方便，我们可以更改`vim`的配置文件，使`vim`的使用更加便捷。我们在`vim ~/.vimrc`写入:
`set ts=4`表示使`Tab`键等效为4个空格`set nu`表示使`vim`显示行号`nu`是`number`缩写

>注2: 在`vim`编辑器中运行`Session()`会话时，有时会出现提示`warning`，是因为有的电脑可以支持加速指令，但是运行代码时并没有启动这些指令。可以把这些提示`warning`暂时屏蔽掉。屏蔽方法为进入主目录下的`bashrc`文件，在`bashrc`文件中加入这样一句`export TF_CPP_MIN_LOG_LEVEL=2`，从而把提示`warning`等级降低。这个命令可以控制`python`程序显示提示信息的等级，在`Tensorflow`里面一般设置成是"0"(显示所有信息)或者"1"(不显示`info`)，"2"代表不显示`warning`，"3"代表不显示 `error`。一般不建议设置成 3。`source`命令用于重新执行修改的初始化文件，使之立即生效，而不必注销并重新登录。


