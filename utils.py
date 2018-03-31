import re
import os
import string
from collections import defaultdict

import numpy as np
from nltk.data import load
from pyvi.pyvi import ViTokenizer

from gensim.models import Word2Vec

VN_SENT_MODEL = 'file:' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'vietnamese.pickle')

def normalize_text(text: str):
    text = re.sub('(?=[^\w|\s+])', ' ', text)
    text = re.sub('(?<=[^\w|\s])(?=\w+|\s+)', ' ', text)
    return re.sub('\s+', ' ', text).strip()


def separate_sentence(paragraph: str):
    tokenizer = load(VN_SENT_MODEL)
    return tokenizer.tokenize(paragraph)


def tokenize_text(sentence: str):
    return ViTokenizer.tokenize(sentence)
    # return word_sent(sentence, format='text')


def remove_punctuation(text: str):
    return " ".join([w for w in text.split() if w not in set(string.punctuation)])


def lowercase(text: str):
    return text.lower()


def create_word_vec(revs: list, dim=300, min_count=5):
    wv = Word2Vec(revs, size=dim, min_count=min_count).wv
    return {w: wv[w] for w in wv.vocab}


def create_vocab(revs):
    vocab = defaultdict(int)
    for rev in revs:
        for word in rev.split():
            vocab[word] += 1
    return vocab


def create_word_idx_map(vocab: list):
    i = 1
    word_idx_map = dict()
    for w in vocab:
        word_idx_map[w] = i
        i += 1
    return word_idx_map


def get_w2v_mat(word_vecs, dim=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    w2v_mat = np.zeros(shape=(vocab_size + 1, dim), dtype='float32')
    w2v_mat[0] = np.zeros(dim, dtype='float32')
    i = 1
    for word in word_vecs:
        w2v_mat[i] = word_vecs[word]
        i += 1
    return w2v_mat


def add_unknown_words(word_vecs, vocab, min_df=1, dim=300):
    """
    For words that occur in at least min_df documents, create a separate word vector. 0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, dim)


def process_vi(text, processors=[normalize_text, lowercase, remove_punctuation, tokenize_text]):
    for processor in processors:
        text = processor(text)
    return text


def read_data_file(file_path):
    revs = []
    revs_idx = []
    revs_content = []

    count_positive = 0
    count_negative = 0

    with open(file_path, encoding='utf-8') as f:
        for line in f.readlines():
            line = re.sub('\n', '', line)
            line = line.split('\t')

            idx, senti, tar, content = int(line[0]), int(line[1]), line[2], line[3]
            
            if idx not in revs_idx:
                revs_idx.append(idx)
                revs_content.append(re.sub('\$t\$', tar, content))

            if senti == 1:
                count_positive += 1
            else:
                count_negative += 1

            revs.append((senti, tar, content))

    print("Positive: {0}".format(count_positive))
    print("Negative: {0}".format(count_negative))

    return revs, revs_content

def creat_word_embedding(revs, word_embeding_dim=100):
    vocab = create_vocab(revs)
    word_idx_map = create_word_idx_map(vocab)
    wv = create_word_vec([rev.split() for rev in revs], dim=word_embeding_dim)
    add_unknown_words(wv, vocab, dim=word_embeding_dim)
    word_embeding = get_w2v_mat(wv, dim=word_embeding_dim)
    return word_idx_map, word_embeding


def data_parse_for_lstm(revs, word_idx_map, max_len):
    x = []
    for rev in revs:
        _, tar, content = rev
        content = content.split()
        content[content.index('$t$')] = tar 
        x.append(list(map(lambda w: word_idx_map.get(w, 0), content)))
    y = label_parse([rev[0] for rev in revs])
    sen_len = [len(sen) for sen in x]
    
    for i in range(len(x)):
        if len(x[i]) < max_len:
            x[i] += [0] * (max_len - len(x[i]))
        else:
            x[i] = x[i][0:max_len]
    return np.asarray(x), np.asarray(sen_len), y


def data_parse(revs, word_idx_map, max_len):
    x_fw, x_bw, target_words = [], [], []
    for rev in revs:
        senti, tar, content = rev
        content = content.split()

        tar_idx = content.index('$t$')

        fw = list(map(lambda w: word_idx_map.get(w, 0), content[0:tar_idx]))
        x_fw.append(fw)
    
        bw = list(map(lambda w: word_idx_map.get(w, 0), content[tar_idx + 1:]))
        bw.reverse()
        x_bw.append(bw)

        target_words.append([word_idx_map.get(tar, 0)])
    y = label_parse([rev[0] for rev in revs])    
    x_fw, len_fw = input_parse(max_len, x_fw, target_words)
    x_bw, len_bw = input_parse(max_len, x_bw, target_words)
    target_words = tar_word_parse(target_words)
    return x_fw, len_fw, x_bw, len_bw, target_words, y


def input_parse(max_len, X, target_words):    
    X = [x + tar for x, tar in zip(X, target_words)]    
    len_x = [len(x) if len(x) < max_len else max_len for x in X]
    for i in range(len(X)):
        if len(X[i]) > max_len:
            X[i] = X[i][len(X[i]) - max_len:]
        else:
            X[i] += [0] * (max_len - len(X[i]))        
    return  np.asarray(X), np.asarray(len_x)


def label_parse(y):
    return np.asarray([[1, 0] if yi == -1 else [0, 1] for yi in y])


def tar_word_parse(target_words):
    return np.asarray(target_words)


def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = list(range(length))
    for j in range(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]
