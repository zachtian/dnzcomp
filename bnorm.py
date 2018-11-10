import sys
import tensorflow as tf
import numpy as np
import argparse

import trainloader as ldr
import testloader as tldr
import model as md
import utils as ut

KEEPLAST = 2
SAVE_FREQ = 1000

BSZ = 96
WEIGHT_DECAY = 0.
MOM = 0.9
IMG_SZ = 32


parser = argparse.ArgumentParser()
parser.add_argument('--DIS', type=int, default=100,
                    help='display frequency')
parser.add_argument('--LIST', type=str, default='data/train.txt',
                    help='txt that contains all the files')
parser.add_argument('--STD', type=str, default='0.0 0.05 0.1 0.15 0.2 0.25',
                    help='list of noise std')
parser.add_argument('--WTS', type=str, default='wts/',
                    help='folder that stores all the params')
parser.add_argument('--ITER', type=int, default=int(1e7),
                    help='total iterations')
args = parser.parse_args()


saved = ut.ckpter(args.WTS + 'iter_*.model.npz')
iter = saved.iter

std_num = len(args.STD.split(' '))
stds = np.zeros(BSZ)

for i in range(std_num):
    stds[i*16:(i+1)*16] = float(args.STD.split(' ')[i])

batcher = ut.batcher(args.LIST, BSZ, iter)

# Set up data prep
data = ldr.trainload(BSZ, IMG_SZ)

# Noise STD
noise_std = tf.placeholder(shape=[BSZ], dtype=tf.float32)

# Load model-def
net = md.model(data.batch, noise_std)

# Learning rate
lr = tf.placeholder(shape=[], dtype=tf.float32)

# Start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if saved.latest is not None:
    sys.stdout.write("Restoring from " + saved.latest + "\n")
    sys.stdout.flush()
    net.load(saved.latest, sess)
    saved.clean(last=KEEPLAST)

# Load first batch
imgs = batcher.get_batch()
_ = sess.run(data.fetchOp, feed_dict=data.getfeed(imgs))

bnorm_mn = []
bnorm_vr = []
# calculate batchnorm
for iter in range(10000):
    # Swap in pre-fetched buffer into current input
    _ = sess.run(data.swapOp)

    # Run training step & getch for next batch
    imgs = batcher.get_batch()
    fdict = data.getfeed(imgs)

    fdict[noise_std] = stds

    mn, vr, _ = sess.run([net.bnorm_mn, net.bnorm_vr, data.fetchOp],
                             feed_dict=fdict)
    temp_bnorm=[]
    for key in mn.keys():
        temp_bnorm.append(mn[key])
    bnorm_mn.append(temp_bnorm)
    temp_bnorm = []
    for key in vr.keys():
        temp_bnorm.append(vr[key])
    bnorm_vr.append(temp_bnorm)

bn_mn = np.average(bnorm_mn, 0)
bn_vr = np.average(bnorm_vr, 0) + np.var(bnorm_mn, 0)

temp = {}
i = 0
for k in mn.keys():
    temp[k] = bn_mn[i]
    i = i+1
np.savez('bnorm/bnorm_mn.npz', **temp)

temp = {}
i = 0
for k in vr.keys():
    temp[k] = bn_vr[i]
    i = i+1
np.savez('bnorm/bnorm_vr.npz', **temp)

