import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import Cnews
import os

tf.reset_default_graph()

TRAINDATA_PATH = '../dataset/cnews/cnews.train.txt'
VALDATA_PATH = '../dataset/cnews/cnews.val.txt'
TESTDATA_PATH = '../dataset/cnews/cnews.val.txt'
AFTER_PREPROCESS_TRAIN = './after_preprocess_train.txt'
STOPWORDS_PATH = './stopwords.txt'
WORD2VEC_MODEL = './w2v.model'


MAX_SEQLEN = 50
BATCH_SIZE = 500
EPOCHES = 10


if __name__ == '__main__':
    if not os.path.exists(AFTER_PREPROCESS_TRAIN):
        x,y = Cnews.cnews_data_preprocess(TRAINDATA_PATH,STOPWORDS_PATH,AFTER_PREPROCESS_TRAIN)
    else:
        x,y = Cnews.read_after_prep_data(AFTER_PREPROCESS_TRAIN)

    x, y = shuffle(x, y)
    num_example = x.shape[0]

    if os.path.exists(WORD2VEC_MODEL):
        w2v = Cnews.get_w2v_model(WORD2VEC_MODEL)
    else:
        w2v = Cnews.train_w2v(WORD2VEC_MODEL,x)
    vec_length = w2v.wv.vector_size

    train_x = tf.placeholder(dtype=tf.float32,shape=[None,MAX_SEQLEN,vec_length,1])
    train_y = tf.placeholder(dtype=tf.int32,shape=[None])

    pool_output = []
    kernel_sizes = [2,2,3,3,4,4]
    for i,k in enumerate(kernel_sizes):
        fmt = 'conv_pool_{}'.format(i)

        with tf.variable_scope(fmt):
            kshape = [k,vec_length,1,1]
            kernel = tf.get_variable('kernel',shape=kshape,dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv = tf.nn.conv2d(train_x,kernel,[1,1,1,1],padding='VALID')
            # print(conv.shape)
            pool = tf.nn.max_pool(conv,[1,MAX_SEQLEN-k+1,1,1],[1,1,1,1],padding='VALID')
            pool_output.append(tf.layers.flatten(pool))
    pool_output = tf.concat([p for p in pool_output],axis=1)

    logit = tf.layers.dense(pool_output,10,activation=None)

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=train_y,logits=logit)
    loss = tf.reduce_mean(cross_entropy)

    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(train_y,tf.int64),tf.argmax(logit,1)),tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(EPOCHES):
            i = 0
            while True:
                start = (i * BATCH_SIZE) % num_example
                end = min(start + BATCH_SIZE,num_example)

                # 获取对应样本的词向量，并控制了序列长度
                sample_x = Cnews.data2wordvec(x[start:end], w2v, max_seqlen=MAX_SEQLEN)
                sample_x = np.array(sample_x).reshape(-1, MAX_SEQLEN, vec_length, 1)
                sample_y = y[start:end]

                feed = {train_x:sample_x, train_y:sample_y}
                _,l,acc = sess.run([train_op,loss,accuracy],feed_dict=feed)
                if end == num_example:
                    break
                i += 1

            # 训练集前5000个样本的训练结果
            sample_x = Cnews.data2wordvec(x[:5000], w2v, max_seqlen=MAX_SEQLEN)
            sample_x = np.array(sample_x).reshape(-1, MAX_SEQLEN, vec_length, 1)
            sample_y = y[:5000]
            feed = {train_x:sample_x,train_y:sample_y}
            test_loss,test_acc = sess.run([loss,accuracy],feed_dict=feed)
            print('epoch:',epoch,' test_loss:',test_loss,' test_acc:',test_acc)

            saver.save(sess,'./model/textcnn.ckpt',epoch)