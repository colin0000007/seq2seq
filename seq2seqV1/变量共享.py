#encoding=UTF-8
import tensorflow as tf

#1.使用variable_scope
with tf.variable_scope("s1") as scope:
    a = tf.get_variable(shape=[1],name='a')
    print(a)

#设置reuse=True达到重用
with tf.variable_scope("s1",reuse=True) as scope:
    b = tf.get_variable(name='a')
    print(b)
    print(a==b)

'''
2.使用name_scope不能达到重用，下面的代码报错
name_scope只是管理一组op，比如一组训练的op，下面的每个变量都会加上前缀
它的作用是让我们区分这些变量属于哪个操作下
with tf.name_scope('training') as scope:
with tf.name_scope('testing') as scope:
'''
'''
with tf.name_scope("ns1") as scope:
    c = tf.get_variable(shape=[1],name='c')
    print(c)
with tf.name_scope("ns1",reuse=True) as scope:
    d = tf.get_variable(shape=[1],name="c")
    print(d)
    print(c==d)
'''
   

