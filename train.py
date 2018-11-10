import sys
import time
import tensorflow as tf
import numpy as np
import argparse

import trainloader as ldr
import trainer as tr
import model as md
import modelv2 as mdv2
import modelv3 as mdv3
import modelv4 as mdv4
import modelv5 as mdv5
import modelv6 as mdv6
import modelv7 as mdv7
import modelv8 as mdv8
import utils as ut
import ctrlc

KEEPLAST = 2
SAVE_FREQ = 1000

BSZ = 176
WEIGHT_DECAY = 0.
MOM = 0.9

parser = argparse.ArgumentParser()
parser.add_argument('--DIS', type=int, default=1000,
                    help='display frequency')
parser.add_argument('--LISTTRAIN', type=str, default='data/train.txt',
                    help='txt that contains all the files')
parser.add_argument('--WTS', type=str, default='wtsv4/',
                    help='folder that stores all the params')
parser.add_argument('--LR', type=float, default=1e-4,
                    help='input learning rate')
parser.add_argument('--STD', type=str, default='0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0',
                    help='list of noise std')
parser.add_argument('--ITER', type=int, default=int(1e8),
                    help='total iterations')
parser.add_argument('--BN', type=str, default='FALSE',
                    help='use bnorm')
parser.add_argument('--MD', type=int, default=4,
                    help='model version')
parser.add_argument('--SIZE', type=int, default=32,
                    help='image size')
parser.add_argument('--LOADADAM', type=str, default='TRUE',
                    help='load adam')
parser.add_argument('--LOSSWEIGHT', type=float, default=0.,
                    help='comb loss')
args = parser.parse_args()

# Check for saved weights
saved = ut.ckpter(args.WTS + 'iter_*.model.npz')
iter = saved.iter

std_num = len(args.STD.split(' '))
stds = np.zeros(BSZ)
s_loss = []

for i in range(std_num):
    s_loss.append([])
    stds[i*16:(i+1)*16] = float(args.STD.split(' ')[i])
# Set up batching
batcher_train = ut.batcher(args.LISTTRAIN, BSZ, iter)


# Set up data prep
data = ldr.trainload(BSZ, args.SIZE)

# Noise STD
std_img = tf.placeholder(shape=[BSZ], dtype=tf.float32)
std_denoise = tf.placeholder(shape=[BSZ], dtype=tf.float32)

# Load model-def
if args.MD == 1:
    net = md.model(data.batch, std_img, std_denoise)
elif args.MD == 2:
    net = mdv2.model(data.batch, std_img, std_denoise)
elif args.MD == 3:
    net = mdv3.model(data.batch, std_img, std_denoise)
elif args.MD == 4:
    net = mdv4.model(data.batch, std_img, std_denoise)
elif args.MD == 5:
    net = mdv5.model(data.batch, std_img, std_denoise)
elif args.MD == 6:
    net = mdv6.model(data.batch, std_img, std_denoise)
elif args.MD == 7:
    net = mdv7.model(data.batch, std_img, std_denoise)
elif args.MD == 8:
    net = mdv8.model(data.batch, std_img, std_denoise)

# Learning rate
lr = tf.placeholder(shape=[], dtype=tf.float32)

# Load trainer-def
opt = tr.train(net, data.batch, lr, WEIGHT_DECAY, args.LOSSWEIGHT)

# Start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Load saved weights if any

if saved.latest is not None:
    sys.stdout.write("Restoring from " + saved.latest + "\n")
    sys.stdout.flush()
    net.load(saved.latest, sess)
    if args.LOADADAM == 'TRUE':
        net.loadAdam(args.WTS + 'adam.npz', opt.optimizer, sess)
    saved.clean(last=KEEPLAST)

# Load first batch
imgs = batcher_train.get_batch()
_ = sess.run(data.fetchOp, feed_dict=data.getfeed(imgs))

while iter < args.ITER and not ctrlc.stop:
    LR = args.LR
    # Swap in pre-fetched buffer into current input
    _ = sess.run(data.swapOp)

    # Run training step & getch for next batch
    imgs = batcher_train.get_batch()
    fdict = data.getfeed(imgs)

    fdict[std_img] = stds
    fdict[std_denoise] = stds
    fdict[lr] = LR

    img_out, loss, outs, _, _ = sess.run([net.out, opt.loss, opt.loss_out, opt.opt, data.fetchOp],
                                      feed_dict=fdict)
    for i in range(std_num):
        s_loss[i].append(outs[i*16:(i+1)*16])
    # Display frequently
    if iter % args.DIS == 0:
        for i in range(std_num):
            loss = np.mean(s_loss[i])
            s_loss[i] = []
            tmstr = time.strftime("%Y-%m-%d %H:%M:%S")
            sys.stdout.write(tmstr + " [%09d] lr=%.2e Train%.2f.loss=%.6f\n"
                            % (iter, LR, stds[i*16], loss))
            sys.stdout.flush()

    iter = iter+1

    # Save periodically
    if iter % SAVE_FREQ == 0:
        fname = args.WTS + "iter_%d.model.npz" % iter
        net.save(fname, sess)
        net.saveAdam(args.WTS + 'adam.npz', opt.optimizer, sess)
        saved.clean(last=KEEPLAST)
        sys.stdout.write("Saved weights to " + fname + "\n")
        sys.stdout.flush()

if saved.iter < iter:
    fname = args.WTS + "iter_%d.model.npz" % iter
    net.save(fname, sess)
    net.saveAdam(args.WTS + 'adam.npz', opt.optimizer, sess)
    saved.clean(last=KEEPLAST)
    sys.stdout.write("Saved weights to " + fname + "\n")
    sys.stdout.flush()
