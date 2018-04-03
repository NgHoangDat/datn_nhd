import tensorflow as tf

from utils import batch_index

def show_res(iter, tr_loss, tr_acc, te_loss, te_acc):
    if iter == 0:
        print('iter\tTrain loss\tTrain acc\tTest loss\tTest acc')
    print('{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(iter, tr_loss, tr_acc, te_loss, te_acc))

class LSTM(object):

    def __init__(self, n_hidden=100, n_class=2, max_sentence_len=50, l2_reg=0.):
        self.word_idx_map = None
        self.w2v = None
        self.embedding_dim = None
        
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.y = tf.placeholder(tf.int32, [None, self.n_class])
            self.sen_len = tf.placeholder(tf.int32, None)

        with tf.name_scope('weights'):
            self.weights = tf.get_variable(
                    name='lstm_w',
                    shape=[self.n_hidden, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.003, 0.003),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )            

        with tf.name_scope('biases'):
            self.biases = tf.get_variable(
                    name='lstm_b',
                    shape=[self.n_class],
                    initializer=tf.random_uniform_initializer(-0.003, 0.003),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )

    def predict(self, x, sen_len, keep_prob=1.0):
        saver = tf.train.Saver()
        inputs = tf.nn.embedding_lookup(self.w2v, self.x)
        prob = tf.nn.softmax(self._predict(inputs))
        with tf.Session() as sess:
            saver.restore(sess, self.save_path)
            return sess.run(prob, {
                self.x: x, 
                self.sen_len: sen_len, 
                self.dropout_keep_prob: keep_prob
            })

    def _predict(self, inputs):
        """
        :params: self.x
        :return: non-norm prediction values
        """

        inputs = tf.nn.dropout(inputs, keep_prob=self.dropout_keep_prob)
        with tf.variable_scope('dynamic_rnn', reuse=tf.AUTO_REUSE):
            outputs, state = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.n_hidden),
                inputs=inputs,
                sequence_length=self.sen_len,
                dtype=tf.float32,
                scope='LSTM'
            )
            batch_size = tf.shape(outputs)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len - 1)
            output = tf.gather(tf.reshape(outputs, [-1, self.n_hidden]), index)  # batch_size * n_hidden
        predict = tf.matmul(output, self.weights) + self.biases

        return predict

    def learn(self, word_idx_map, w2v, train_data, test_data, save_path, n_iter=100, batch_size=64, learning_rate=0.01):
        self.word_idx_map = word_idx_map
        self.w2v = tf.Variable(w2v, name='word_embedding')
        self.embedding_dim = w2v.shape[1]

        inputs = tf.nn.embedding_lookup(self.w2v, self.x)
        prob = self._predict(inputs)

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prob, labels=self.y))

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.int32))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            tr_x, tr_sen_len, tr_y = train_data
            te_x, te_sen_len, te_y = test_data

            init = tf.global_variables_initializer()
            sess.run(init)

            max_acc = 0.
            def test(x, y, sen_len, keep_prob):
                acc, loss = sess.run([accuracy, cost], feed_dict={
                    self.x: x,
                    self.y: y,
                    self.sen_len: sen_len,
                    self.dropout_keep_prob: keep_prob
                })
                return acc / y.shape[0], loss / y.shape[0]

            tr_acc, tr_loss = test(tr_x, tr_y, tr_sen_len, 1.0)
            te_acc, te_loss = test(te_x, te_y, te_sen_len, 1.0)
            show_res(0, tr_loss, tr_acc, te_loss, te_acc)

            for i in range(n_iter):
                for train, _ in self.get_batch_data(tr_x, tr_y, tr_sen_len, batch_size, 0.5):
                    sess.run([optimizer, global_step], feed_dict=train)
                tr_acc, tr_loss = test(tr_x, tr_y, tr_sen_len, 1.0)
                te_acc, te_loss = test(te_x, te_y, te_sen_len, 1.0)
                show_res(i + 1, tr_loss, tr_acc, te_loss, te_acc)

                if te_acc > max_acc:
                    max_acc = te_acc
                    self.save_path = saver.save(sess, save_path)
            print('Optimization Finished! Max acc={}'.format(max_acc))
            

    def get_batch_data(self, x, y, sen_len, batch_size, keep_prob):
        for index in batch_index(len(y), batch_size, 1):
            feed_dict = {
                self.x: x[index],
                self.y: y[index],
                self.sen_len: sen_len[index],
                self.dropout_keep_prob: keep_prob,
            }
            yield feed_dict, len(index)


class TC_LSTM(object):

    def __init__(self, n_hidden=100, n_class=2, max_sentence_len=50, l2_reg=0.):
        self.word_idx_map = None
        self.w2v = None
        self.embedding_dim = None

        self.n_hidden = n_hidden
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg

        self.dropout_keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope('inputs'):
            self.x_fw = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.sen_len_fw = tf.placeholder(tf.int32, None)

            self.x_bw = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.sen_len_bw = tf.placeholder(tf.int32, [None])

            self.target_words = tf.placeholder(tf.int32, [None, 1])
            self.y = tf.placeholder(tf.int32, [None, self.n_class])

        with tf.name_scope('weights'):
            self.weights = tf.get_variable(
                    name='tc_bi_lstm_w',
                    shape=[2 * self.n_hidden, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.003, 0.003),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )

        with tf.name_scope('biases'):
            self.biases = tf.get_variable(
                    name='tc_bi_lstm_b',
                    shape=[self.n_class],
                    initializer=tf.random_uniform_initializer(-0.003, 0.003),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )

    def predict(self, x_fw, len_fw, x_bw, len_bw, target_words, keep_prob=1.0):
        saver = tf.train.Saver()
        inputs_fw = tf.nn.embedding_lookup(self.w2v, self.x_fw)
        inputs_bw = tf.nn.embedding_lookup(self.w2v, self.x_bw)
        
        target = tf.reduce_mean(tf.nn.embedding_lookup(self.w2v, self.target_words), 1, keepdims=True)
        _batch_size = tf.shape(inputs_bw)[0]
        target = tf.zeros([_batch_size, self.max_sentence_len, self.embedding_dim]) + target
        
        inputs_fw = tf.concat([inputs_fw, target], 2)
        inputs_bw = tf.concat([inputs_bw, target], 2)
        
        prob = tf.nn.softmax(self._predict(inputs_fw, inputs_bw))
        with tf.Session() as sess:
            saver.restore(sess, self.save_path)
            return sess.run(prob, {
                self.x_fw: x_fw,
                self.x_bw: x_bw,
                self.target_words: target_words,
                self.sen_len_fw: len_fw,
                self.sen_len_bw: len_bw,
                self.dropout_keep_prob: keep_prob
            })

    def _predict(self, inputs_fw, inputs_bw):
        """
        :params: 
            inputs_fw: sequence of words to target word in forward direction
            inputs_bw: sequence of words to target word in backward direction
        :return: non-norm prediction values
        """
        
        inputs_fw = tf.nn.dropout(inputs_fw, keep_prob=self.dropout_keep_prob)
        inputs_bw = tf.nn.dropout(inputs_bw, keep_prob=self.dropout_keep_prob)

        with tf.variable_scope('forward_lstm', reuse=tf.AUTO_REUSE):
            outputs_fw, state_fw = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.n_hidden),
                inputs=inputs_fw,
                sequence_length=self.sen_len_fw,
                dtype=tf.float32,
                scope='LSTM_fw'
            )
            batch_size = tf.shape(outputs_fw)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len_fw - 1)
            output_fw = tf.gather(tf.reshape(outputs_fw, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        with tf.variable_scope('backward_lstm', reuse=tf.AUTO_REUSE):
            outputs_bw, state_bw = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.n_hidden),
                inputs=inputs_bw,
                sequence_length=self.sen_len_bw,
                dtype=tf.float32,
                scope='LSTM_bw'
            )
            batch_size = tf.shape(outputs_bw)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len_bw - 1)
            output_bw = tf.gather(tf.reshape(outputs_bw, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        output = tf.concat([output_fw, output_bw], 1)  # batch_size * 2n_hidden
        predict = tf.matmul(output, self.weights) + self.biases
        return predict

    def learn(self, word_idx_map, w2v, train_data, test_data, save_path, n_iter=100, batch_size=64, learning_rate=0.01):
        """[summary]
        
        [description]
        
        Arguments:
            word_idx_map {[type]} -- [description]
            w2v {[type]} -- [description]
            train_data {[type]} -- x_fw, x_bw, tar_w, y
            test_data {[type]} -- x_fw, x_bw, tar_w, y
        """
        self.word_idx_map = word_idx_map
        self.w2v = tf.Variable(w2v, name='word_embedding')
        self.embedding_dim = w2v.shape[1]
        
        inputs_fw = tf.nn.embedding_lookup(self.w2v, self.x_fw)
        inputs_bw = tf.nn.embedding_lookup(self.w2v, self.x_bw)
        
        target = tf.reduce_mean(tf.nn.embedding_lookup(self.w2v, self.target_words), 1, keepdims=True)
        _batch_size = tf.shape(inputs_bw)[0]
        target = tf.zeros([_batch_size, self.max_sentence_len, self.embedding_dim]) + target
        
        inputs_fw = tf.concat([inputs_fw, target], 2)
        inputs_bw = tf.concat([inputs_bw, target], 2)
        
        prob = self._predict(inputs_fw, inputs_bw)

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prob, labels=self.y))

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
        saver = tf.train.Saver()
        with tf.Session() as sess:

            tr_x_fw, tr_len_fw, tr_x_bw, tr_len_bw, tr_target_word, tr_y = train_data
            te_x_fw, te_len_fw, te_x_bw, te_len_bw, te_target_word, te_y = test_data

            init = tf.global_variables_initializer()
            sess.run(init)

            max_acc = 0.

            def test(x_fw, len_fw, x_bw, len_bw, y, target_words, keep_prob):
                acc, loss = sess.run([accuracy, cost], feed_dict={
                    self.x_fw: x_fw, self.x_bw: x_bw, self.y: y,
                    self.sen_len_fw: len_fw,
                    self.sen_len_bw: len_bw,
                    self.target_words: target_words,
                    self.dropout_keep_prob: keep_prob
                })
                return acc / y.shape[0], loss / y.shape[0]

            tr_acc, tr_loss = test(tr_x_fw, tr_len_fw, tr_x_bw, tr_len_bw, tr_y, tr_target_word, 1.0)
            te_acc, te_loss = test(te_x_fw, te_len_fw, te_x_bw, te_len_bw, te_y, te_target_word, 1.0)

            show_res(0, tr_loss, tr_acc, te_loss, te_acc)

            for i in range(n_iter):
                for train, _ in self.get_batch_data(tr_x_fw, tr_len_fw, tr_x_bw, tr_len_bw, tr_y, tr_target_word, batch_size, 0.5):
                    sess.run([optimizer, global_step], feed_dict=train)
             
                tr_acc, tr_loss = test(tr_x_fw, tr_len_fw, tr_x_bw, tr_len_bw, tr_y, tr_target_word, 1.0)
                te_acc, te_loss = test(te_x_fw, te_len_fw, te_x_bw, te_len_bw, te_y, te_target_word, 1.0)

                show_res(i + 1, tr_loss, tr_acc, te_loss, te_acc)

                if te_acc > max_acc:
                    max_acc = te_acc
                    self.save_path = saver.save(sess, save_path)
            print('Optimization Finished! Max acc={}'.format(max_acc))


    def get_batch_data(self, x, sen_len, x_bw, sen_len_bw, y, target_words, batch_size, keep_prob):
        for index in batch_index(len(y), batch_size, 1):
            feed_dict = {
                self.x_fw: x[index],
                self.x_bw: x_bw[index],
                self.y: y[index],
                self.sen_len_fw: sen_len[index],
                self.sen_len_bw: sen_len_bw[index],
                self.target_words: target_words[index],
                self.dropout_keep_prob: keep_prob,
            }
            yield feed_dict, len(index)


class TD_LSTM(object):

    def __init__(self, n_hidden=100, n_class=2, max_sentence_len=50, l2_reg=0.):
        self.word_idx_map = None
        self.w2v = None
        self.embedding_dim = None

        self.n_hidden = n_hidden
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope('inputs'):
            self.x_fw = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.len_fw = tf.placeholder(tf.int32, None)

            self.x_bw = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.len_bw = tf.placeholder(tf.int32, [None])

            self.y = tf.placeholder(tf.int32, [None, self.n_class])
            
        with tf.name_scope('weights'):
            self.weights = tf.get_variable(
                name='td_bi_lstm_w',
                shape=[2 * self.n_hidden, self.n_class],
                initializer=tf.random_uniform_initializer(-0.003, 0.003),
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
            )

        with tf.name_scope('biases'):
            self.biases = tf.get_variable(
                name='td_bi_lstm_b',
                shape=[self.n_class],
                initializer=tf.random_uniform_initializer(-0.003, 0.003),
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
            )

    def predict(self, x_fw, len_fw, x_bw, len_bw, keep_prob=1.0):
        saver = tf.train.Saver()
        inputs_fw = tf.nn.embedding_lookup(self.w2v, self.x_fw)
        inputs_bw = tf.nn.embedding_lookup(self.w2v, self.x_bw)
        prob = tf.nn.softmax(self._predict(inputs_fw, inputs_bw))
        with tf.Session() as sess:
            saver.restore(sess, self.save_path)
            return sess.run(prob, {
                self.x_fw: x_fw,
                self.len_fw: len_fw,
                self.x_bw: x_bw,
                self.len_bw: len_bw,
                self.dropout_keep_prob: keep_prob
            })

    def _predict(self, inputs_fw, inputs_bw):
        """
        :params: self.x_fw, self.x_bw, self.seq_len, self.seq_len_bw,
                self.weights['softmax_lstm'], self.biases['softmax_lstm']
        :return: non-norm prediction values
        """
        inputs_fw = tf.nn.dropout(inputs_fw, keep_prob=self.dropout_keep_prob)
        inputs_bw = tf.nn.dropout(inputs_bw, keep_prob=self.dropout_keep_prob)

        with tf.variable_scope('forward_lstm', reuse=tf.AUTO_REUSE):
            outputs_fw, state_fw = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.n_hidden),
                inputs=inputs_fw,
                sequence_length=self.len_fw,
                dtype=tf.float32,
                scope='LSTM_fw'
            )
            batch_size = tf.shape(outputs_fw)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (self.len_fw - 1)
            output_fw = tf.gather(tf.reshape(outputs_fw, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        with tf.variable_scope('backward_lstm', reuse=tf.AUTO_REUSE):
            outputs_bw, state_bw = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.n_hidden),
                inputs=inputs_bw,
                sequence_length=self.len_bw,
                dtype=tf.float32,
                scope='LSTM_bw'
            )
            batch_size = tf.shape(outputs_bw)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (self.len_bw - 1)
            output_bw = tf.gather(tf.reshape(outputs_bw, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        output = tf.concat([output_fw, output_bw], 1)  # batch_size * 2n_hidden
        predict = tf.matmul(output, self.weights) + self.biases
        return predict

    def learn(
        self, word_idx_map, w2v, train_data, test_data, save_path,
        n_iter=100, batch_size=64, learning_rate=0.01
    ):
        self.word_idx_map = word_idx_map
        self.w2v = tf.Variable(w2v, name='word_embedding')
        self.embedding_dim = w2v.shape[1]

        inputs_fw = tf.nn.embedding_lookup(self.w2v, self.x_fw)
        inputs_bw = tf.nn.embedding_lookup(self.w2v, self.x_bw)
        prob = self._predict(inputs_fw, inputs_bw)

        with tf.name_scope('loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prob, labels=self.y)) + sum(reg_loss)

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
        saver = tf.train.Saver()
        with tf.Session() as sess:
 
            tr_x_fw, tr_len_fw, tr_x_bw, tr_len_bw, _, tr_y = train_data
            te_x_fw, te_len_fw, te_x_bw, te_len_bw, _, te_y = test_data

            init = tf.global_variables_initializer()
            sess.run(init)

            max_acc = 0.

            def test(x_fw, len_fw, x_bw, len_bw, y, keep_prob):
                acc, loss = sess.run([accuracy, cost], {
                    self.x_fw: x_fw, self.x_bw: x_bw, self.y: y,
                    self.len_fw: len_fw,
                    self.len_bw: len_bw,
                    self.dropout_keep_prob: keep_prob
                })
                return acc / y.shape[0], loss / y.shape[0]

            tr_acc, tr_loss = test(tr_x_fw, tr_len_fw, tr_x_bw, tr_len_bw, tr_y, 1.0)
            te_acc, te_loss = test(te_x_fw, te_len_fw, te_x_bw, te_len_bw, te_y, 1.0)
            show_res(0, tr_loss, tr_acc, te_loss, te_acc)

            for i in range(n_iter):
                for train, _ in self.get_batch_data(tr_x_fw, tr_len_fw, tr_x_bw, tr_len_bw, tr_y, batch_size, 0.5):
                    sess.run([optimizer, global_step], feed_dict=train)

                tr_acc, tr_loss = test(tr_x_fw, tr_len_fw, tr_x_bw, tr_len_bw, tr_y, 1.0)
                te_acc, te_loss = test(te_x_fw, te_len_fw, te_x_bw, te_len_bw, te_y, 1.0)
                show_res(i + 1, tr_loss, tr_acc, te_loss, te_acc)

                if te_acc > max_acc:
                    max_acc = te_acc
                    self.save_path = saver.save(sess, save_path)

            print('Optimization Finished! Max acc={}'.format(max_acc))

    def get_batch_data(self, x_fw, len_fw, x_bw, len_bw, y, batch_size, keep_prob):
        for index in batch_index(len(y), batch_size, 1):
            feed_dict = {
                self.x_fw: x_fw[index],
                self.x_bw: x_bw[index],
                self.y: y[index],
                self.len_fw: len_fw[index],
                self.len_bw: len_bw[index],
                self.dropout_keep_prob: keep_prob
            }
            yield feed_dict, len(index)
