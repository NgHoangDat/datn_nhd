import tensorflow as tf

from utils import batch_index

def show_res(iter, tr_loss, tr_acc, te_loss, te_acc):
    if iter == 0:
        print('iter\tTrain loss\tTrain acc\tTest loss\tTest acc')
    print('{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(iter, tr_loss, tr_acc, te_loss, te_acc))

class Text_CNN(object):
    """docstring for CNN"""
    def __init__(
        self, 
        max_len=41, n_classes=2, 
        filter_hs=[3, 5, 7], n_filters=128, 
        l2_reg_lambda=0.0
    ):
        self.word_idx_map = None
        self.word_embedding = None
        self.embedding_dim = None
        self.pooled_outputs = None

        self.max_len = max_len
        self.n_classes = n_classes
        self.filter_hs = filter_hs
        self.n_filters = n_filters
        
        self.l2_reg_lambda = l2_reg_lambda
        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, max_len])
            self.y = tf.placeholder(tf.float32, [None, n_classes])

    def predict(self, input):        
        
        if not (self.pooled_outputs):
            raise Exception('The model needs training first')

        n_filters_total = self.n_filters * len(self.filter_hs)
        h_pool = tf.concat(self.pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, n_filters_total])
        l2_loss = tf.constant(0.0)

        with tf.name_scope("dropout"):
           h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        
        with tf.name_scope("output"):

            W = tf.get_variable(
                "W",
                shape=[n_filters_total, self.n_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[self.n_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
            predictions = tf.argmax(scores, 1, name="predictions")
        return scores, predictions, l2_loss

    def learn(
        self, word_idx_map, word_embedding, train_data, test_data, 
        n_iters=10, batch_size=64, learning_rate=1e-3
    ):

        self.word_idx_map = word_idx_map
        self.word_embedding = tf.Variable(word_embedding)
        self.embedding_dim = word_embedding.shape[1]

        with tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([
                    len(self.word_idx_map) + 1, 
                    self.embedding_dim
                ], -1.0, 1.0, name='W')
            )
            embedded_chars_expanded = tf.expand_dims(
                tf.nn.embedding_lookup(W, self.x), -1
            )
        
        self.pooled_outputs = []
        for i, filter_h in enumerate(self.filter_hs):
            with tf.name_scope("conv-maxpool"):
                filter_shape = [filter_h, self.embedding_dim, 1, self.n_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.n_filters]), name='b')
                conv = tf.nn.conv2d(
                    embedded_chars_expanded, W,
                    strides=[1] * 4,
                    padding='VALID',
                    name='conv'
                )
                pooled = tf.nn.max_pool(
                    tf.nn.relu(tf.nn.bias_add(conv, b), name="relu"),
                    ksize=[1, self.max_len - filter_h + 1, 1, 1],
                    strides=[1] * 4,
                    padding='VALID',
                    name='pool'
                )
                self.pooled_outputs.append(pooled)

        inputs = tf.nn.embedding_lookup(self.word_embedding, self.x)
        scores, predictions, l2_loss = self.predict(inputs)

        with tf.name_scope("loss"):
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores, labels=self.y)
            ) + self.l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_sum(
                tf.cast(tf.equal(predictions, tf.argmax(self.y, 1)
            ), "float"), name="accuracy")

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate
            ).minimize(cost, global_step=global_step)

        with tf.Session() as sess:
            tr_x, tr_y = train_data
            te_x, te_y = test_data

            init = tf.global_variables_initializer()
            sess.run(init)

            max_acc = 0.0
            def test(x, y, batch_size, keep_prob):
                acc, loss, cnt = 0., 0., 0.
                for test, num in self.get_batch_data(x, y, batch_size, keep_prob):
                    _loss, _acc = sess.run([cost, accuracy], feed_dict=test)
                    acc += _acc
                    loss += _loss * num
                    cnt += num
                return acc / cnt, loss / cnt

            tr_acc, tr_loss = test(tr_x, tr_y, 1000, 0.5)
            te_acc, te_loss = test(te_x, te_y, 1000, 0.5)
            show_res(0, tr_loss, tr_acc, te_loss, te_acc)
            
            for i in range(n_iters):
                for train, _ in self.get_batch_data(tr_x, tr_y, batch_size, 1.0):
                    sess.run([optimizer, global_step], feed_dict=train)
                
                tr_acc, tr_loss = test(tr_x, tr_y, 1000, 0.5)
                te_acc, te_loss = test(te_x, te_y, 1000, 0.5)
                show_res(i + 1, tr_loss, tr_acc, te_loss, te_acc)

                if te_acc > max_acc:
                    max_acc = te_acc

            print('Optimization Finished! Max acc={}'.format(max_acc))

    def get_batch_data(self, x, y, batch_size, keep_prob):
        for index in batch_index(len(y), batch_size, 1):
            feed_dict = {
                self.x: x[index],
                self.y: y[index],
                self.dropout_keep_prob: keep_prob,
            }
            yield feed_dict, len(index)
