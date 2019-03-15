import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import Cnews
import os


TESTDATA_PATH = '../dataset/cnews/cnews.val.txt'
AFTER_PREPROCESS_TEST = './after_preprocess_test.txt'
STOPWORDS_PATH = './stopwords.txt'
WORD2VEC_MODEL = './w2v.model'

MAX_SEQLEN = 50

if not os.path.exists(AFTER_PREPROCESS_TEST):
    x, y = Cnews.cnews_data_preprocess(TESTDATA_PATH, STOPWORDS_PATH, AFTER_PREPROCESS_TEST)
else:
    x, y = Cnews.read_after_prep_data(AFTER_PREPROCESS_TEST)

w2v = Cnews.get_w2v_model(WORD2VEC_MODEL)
vec_length = w2v.wv.vector_size

x, y = shuffle(x, y)
x = Cnews.data2wordvec(x, w2v,max_seqlen=MAX_SEQLEN)

x = np.array(x).reshape(-1,MAX_SEQLEN,vec_length,1)
num_example = x.shape[0]

test_x = tf.placeholder(dtype=tf.float32,shape=[None,MAX_SEQLEN,vec_length,1])
test_y = tf.placeholder(dtype=tf.int32,shape=[None])

pool_output = []
kernel_sizes = [2,2,3,3,4,4]
for i,k in enumerate(kernel_sizes):
    fmt = 'conv_pool_{}'.format(i)

    with tf.variable_scope(fmt):
        kshape = [k,vec_length,1,1]
        kernel = tf.get_variable('kernel',shape=kshape,dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(test_x,kernel,[1,1,1,1],padding='VALID')
        pool = tf.nn.max_pool(conv,[1,MAX_SEQLEN-k+1,1,1],[1,1,1,1],padding='VALID')
        pool_output.append(tf.layers.flatten(pool))

pool_output = tf.concat([p for p in pool_output],axis=1)

logit = tf.layers.dense(pool_output,10,activation=None)

loss = tf.losses.sparse_softmax_cross_entropy(labels=test_y,logits=logit)
loss_mean = tf.reduce_mean(loss)

pred_y = tf.argmax(logit,1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(test_y,tf.int64),pred_y),tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
        print('get model:',ckpt.model_checkpoint_path)

        feed = {test_x:x,test_y:y}
        test_loss,test_acc,pred = sess.run([loss_mean,accuracy,pred_y],feed_dict=feed)
        print('test_loss:',test_loss,'test_acc:',test_acc)
        print(classification_report(y,pred))
    else:
        print('not fount model.')