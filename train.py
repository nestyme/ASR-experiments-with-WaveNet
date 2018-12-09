import sugartensor as tf
from utils import SpeechCorpus, voca_size



num_blocks = 3     
num_dim = 128      

def get_logit(x, voca_size):

    def res_block(tensor, size, rate, block, dim=num_dim):

        with tf.sg_context(name='block_%d_%d' % (block, rate)):
            conv_filter = tensor.sg_aconv1d(size=size, rate=rate, act='tanh', bn=True, name='conv_filter')

            conv_gate = tensor.sg_aconv1d(size=size, rate=rate,  act='sigmoid', bn=True, name='conv_gate')
            out = conv_filter * conv_gate
            out = out.sg_conv1d(size=1, dim=dim, act='tanh', bn=True, name='conv_out')
            return out + tensor, out
    with tf.sg_context(name='front'):
        z = x.sg_conv1d(size=1, dim=num_dim, act='tanh', bn=True, name='conv_in')
    skip = 0  
    for i in range(num_blocks):
        for r in [1, 2, 4, 8, 16]:
            z, s = res_block(z, size=7, rate=r, block=i)
            skip += s
    with tf.sg_context(name='logit'):
        logit = (skip
                 .sg_conv1d(size=1, act='tanh', bn=True, name='conv_1')
                 .sg_conv1d(size=1, dim=voca_size, name='conv_2'))

    return logit


batch_size = 16    

data = SpeechCorpus(batch_size=batch_size * tf.sg_gpus())
inputs = tf.split(data.mfcc, tf.sg_gpus(), axis=0)
labels = tf.split(data.label, tf.sg_gpus(), axis=0)
seq_len = []
for input_ in inputs:
    seq_len.append(tf.not_equal(input_.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1))

@tf.sg_parallel
def get_loss(opt):
    logit = get_logit(opt.input[opt.gpu_index], voca_size=voca_size)
    return logit.sg_ctc(target=opt.target[opt.gpu_index], seq_len=opt.seq_len[opt.gpu_index])


tf.sg_train(lr=0.0001, loss=get_loss(input=inputs, target=labels, seq_len=seq_len),
            ep_size=data.num_batch, max_ep=50)
