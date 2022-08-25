import tensorflow.compat.v1 as tf
import pandas as pd

tf.disable_v2_behavior()

df = pd.read_csv("hydroparam.csv", header=None)
dataset = df.values

B_tip = dataset[:, 0]
B_70 = dataset[:, 1]
A_tip = dataset[:, 2]
A_70 = dataset[:, 3]
Cp = dataset[:, 4]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
x4 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
w4 = tf.Variable(tf.random_normal([1]), name='weight4')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4 + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):

    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],

                                   feed_dict={x1: B_tip, x2: B_70, x3: A_tip, x4: A_70, Y : Cp})

    if step % 2000 == 0:
        print(step, "Cost : ", cost_val, "\nPrediction:\n", hy_val,

              "\nWeight,bias :\n", sess.run(w1), sess.run(w2), sess.run(w3), sess.run(w4), sess.run(b))

Cp_OPT = 1.306*sess.run(w1)+1.129*sess.run(w2)+1.109*sess.run(w3)+0.489*sess.run(w4)+sess.run(b)
print(Cp_OPT)