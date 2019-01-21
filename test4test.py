import tensorflow as tf
import os

ckpt_dir = "./model/"

if os.path.exists(ckpt_dir):
    epoch = tf.Variable(0, name='epoch', trainable=False)
else:
    epoch = tf.Variable(1, name='epoch', trainable=False)

saver = tf.train.Saver(max_to_keep=5)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


ckpt = tf.train.latest_checkpoint(ckpt_dir)
print(ckpt)
if ckpt != None:
    print("restore")
    saver.restore(sess, ckpt)
else:
    print('Train from scratch')
    saver.save(sess,ckpt_dir+"mmm")

start = sess.run(epoch)
print(start)