import sugartensor as tf
import numpy as np
import pandas as pd
import csv
import string

Import librosa


index2byte = ['<EMP>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


byte2index = {}
for i, ch in enumerate(index2byte):
    byte2index[ch] = i


voca_size = len(index2byte)



def str2index(str_):

    str_ = ' '.join(str_.split())
    str_ = str_.translate(None, string.punctuation).lower()

    res = []
    for ch in str_:
        try:
            res.append(byte2index[ch])
        except KeyError:
            pass
    return res


def index2str(index_list):
    str_ = ''
    for ch in index_list:
        if ch > 0:
            str_ += index2byte[ch]
        elif ch == 0:  # <EOS>
            break
    return str_


def print_index(indices):
    c=''
    for index_list in indices:
        c+=index2str(index_list)
    return c


def _load_mfcc(src_list):

    label, mfcc_file = src_list
    label = np.fromstring(label, np.int)
    mfcc = np.load(mfcc_file, allow_pickle=False)
    mfcc = _augment_speech(mfcc)
    return label, mfcc


def _augment_speech(mfcc):

    r = np.random.randint(-2, 2)

    mfcc = np.roll(mfcc, r, axis=0)

    if r > 0:
        mfcc[:r, :] = 0
    elif r < 0:
        mfcc[r:, :] = 0

    return mfcc


class SpeechCorpus(object):

    def __init__(self, batch_size=16, set_name='train'):

        label, mfcc_file = [], []
        with open(_data_path + 'preprocess/meta/%s.csv' % set_name) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                mfcc_file.append(_data_path + 'preprocess/mfcc/' + row[0] + '.npy')
                label.append(np.asarray(row[1:], dtype=np.int).tostring())

        label_t = tf.convert_to_tensor(label)
        mfcc_file_t = tf.convert_to_tensor(mfcc_file)

        label_q, mfcc_file_q \
            = tf.train.slice_input_producer([label_t, mfcc_file_t], shuffle=True)

        label_q, mfcc_q = _load_mfcc(source=[label_q, mfcc_file_q],
                                     dtypes=[tf.sg_intx, tf.sg_floatx],
                                     capacity=256, num_threads=64)

        batch_queue = tf.train.batch([label_q, mfcc_q], batch_size,
                                     shapes=[(None,), (20, None)],
                                     num_threads=64, capacity=batch_size*32,
                                     dynamic_pad=True)
        self.label, self.mfcc = batch_queue
        # batch * time * dim
        self.mfcc = self.mfcc.sg_transpose(perm=[0, 2, 1])
        self.num_batch = len(label) // batch_size
