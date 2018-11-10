import sys
import tensorflow as tf
import numpy as np
import argparse
import csv

import testloader as tldr
import model as md
import modelv2 as mdv2
import modelv3 as mdv3
import modelv4 as mdv4
import modelv5 as mdv5
import modelv6 as mdv6
import modelv7 as mdv7
import utils as ut

KEEPLAST = 2
SAVE_FREQ = 1000

WEIGHT_DECAY = 0.
MOM = 0.9


parser = argparse.ArgumentParser()
parser.add_argument('--LIST', type=str, default='data/train.txt',
                    help='txt that contains all the files')
parser.add_argument('--STD', type=str, default='0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0',
                    help='list of noise std')
parser.add_argument('--WTS', type=str, default='wtsv2/',
                    help='folder that stores all the params')
parser.add_argument('--ITER', type=int, default=int(1e7),
                    help='total iterations')
parser.add_argument('--BN', type=str, default='False',
                    help='use bnorm')
parser.add_argument('--MD', type=int, default=2,
                    help='model version')
parser.add_argument('--SIZE', type=int, default=36,
                    help='image size')
parser.add_argument('--BSIZE', type=int, default=8,
                    help='batch size')
parser.add_argument('--LOSSWEIGHT', type=float, default=0.,
                    help='comb loss')
args = parser.parse_args()
saved = ut.ckpter(args.WTS + 'iter_*.model.npz')
iter = saved.iter

std_num = len(args.STD.split(' '))
stds = [float(i) for i in args.STD.split(' ')]

# Start session
sess = tf.Session()

batcher = ut.batcher(args.LIST, 1)
data = tldr.testload(args.SIZE, args.BSIZE)
std_img = tf.placeholder(shape=[args.BSIZE*args.BSIZE], dtype=tf.float32)
std_denoise = tf.placeholder(shape=[args.BSIZE*args.BSIZE], dtype=tf.float32)
if args.MD == 1:
    net = md.model(data.batch, std_img, std_denoise, ifbn=(args.BN == 'TRUE'), inp_bnorm=(args.BN == 'TRUE'))
elif args.MD == 2:
    net = mdv2.model(data.batch, std_img, std_denoise, ifbn=(args.BN == 'TRUE'), inp_bnorm=(args.BN == 'TRUE'))
elif args.MD == 3:
    net = mdv3.model(data.batch, std_img, std_denoise, ifbn=(args.BN == 'TRUE'), inp_bnorm=(args.BN == 'TRUE'))
elif args.MD == 4:
    net = mdv4.model(data.batch, std_img, std_denoise, ifbn=(args.BN == 'TRUE'), inp_bnorm=(args.BN == 'TRUE'))
elif args.MD == 5:
    net = mdv5.model(data.batch, std_img, std_denoise, ifbn=(args.BN == 'TRUE'), inp_bnorm=(args.BN == 'TRUE'))
elif args.MD == 6:
    net = mdv6.model(data.batch, std_img, std_denoise, ifbn=(args.BN == 'TRUE'), inp_bnorm=(args.BN == 'TRUE'))
elif args.MD == 7:
    net = mdv7.model(data.batch, std_img, std_denoise, ifbn=(args.BN == 'TRUE'), inp_bnorm=(args.BN == 'TRUE'))

sess.run(tf.global_variables_initializer())

if saved.latest is not None:
    sys.stdout.write("Restoring from " + saved.latest + "\n")
    sys.stdout.flush()
    net.load(saved.latest, sess)
    saved.clean(last=KEEPLAST)

imgs = batcher.get_batch()
_ = sess.run(data.fetchOp, feed_dict=data.getfeed(imgs))
img_size = args.SIZE*args.BSIZE
table = np.zeros([std_num, 3])
for i in range(std_num):
    table[i][0] = stds[i]*10
for i in range(10):
    _ = sess.run(data.swapOp)
    imgs = batcher.get_batch()
    fdict = data.getfeed(imgs)
    fdict[std_img] = np.repeat(stds[0], args.BSIZE*args.BSIZE)
    # Run training step & getch for next batch
    for k in range(std_num):
        fdict[std_denoise] = np.repeat(stds[k], args.BSIZE*args.BSIZE)

        loss, repre,  _ = sess.run([net.loss, net.rep, data.fetchOp],
                                                feed_dict=fdict)
        num_non_zero = 0
        for repre_i in repre:
            num_non_zero = num_non_zero + np.count_nonzero(repre_i)
        table[k][1] = table[k][1] + num_non_zero / (img_size*img_size*3.0)
        table[k][2] = table[k][2] + loss
np.savetxt('rep/modelv'+str(args.MD)+'l'+str(args.LOSSWEIGHT)+'.csv', table/10., delimiter=",")

