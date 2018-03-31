import tensorflow as tf

from utils import read_data_file, creat_word_embedding, data_parse
from LSTM import TD_LSTM

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 100, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 64, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 200, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 20, 'max number of tokens per sentence')

tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_integer('n_iter', 10, 'number of train iter')

tf.app.flags.DEFINE_string('train_file_path', 'data/restaurant/train.txt', 'training file')
tf.app.flags.DEFINE_string('test_file_path', 'data/restaurant/test.txt', 'testing file')

def main(_):
    print('loading data...')
    tr_revs, tr_revs_content = read_data_file(FLAGS.train_file_path)
    word_idx_map, w2v = creat_word_embedding(tr_revs_content, FLAGS.embedding_dim)

    tr_data = data_parse(tr_revs, word_idx_map, FLAGS.max_sentence_len)

    te_revs, _ = read_data_file(FLAGS.test_file_path)
    te_data = data_parse(te_revs, word_idx_map, FLAGS.max_sentence_len)

    td_lstm = TD_LSTM(
        batch_size=FLAGS.batch_size,
        n_hidden=FLAGS.n_hidden,
        learning_rate=FLAGS.learning_rate,
        n_class=FLAGS.n_class,
        max_sentence_len=FLAGS.max_sentence_len,
        l2_reg=FLAGS.l2_reg,
        n_iter=FLAGS.n_iter,
    )
    print('start training...')
    td_lstm.train(word_idx_map, w2v, tr_data, te_data)

if __name__ == '__main__':
    tf.app.run()