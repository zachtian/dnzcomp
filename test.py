import sys
import tensorflow as tf
import numpy as np
import argparse
from PIL import Image, ImageDraw

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
parser.add_argument('--LIST', type=str, default='data/test.txt',
                    help='txt that contains all the files')
parser.add_argument('--STD', type=str, default='0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0',
                    help='list of noise std')
parser.add_argument('--WTS', type=str, default='wtsv2/',
                    help='folder that stores all the params')
parser.add_argument('--BN', type=str, default='FALSE',
                    help='use bnorm')
parser.add_argument('--MD', type=int, default=2,
                    help='model version')
parser.add_argument('--ITER', type=int, default=10,
                    help='total iterations')
parser.add_argument('--SIZE', type=int, default=36,
                    help='image size')
parser.add_argument('--BSIZE', type=int, default=8,
                    help='batch size')
parser.add_argument('--IMGPATH', type=str, default='imgout/',
                    help='image path')
parser.add_argument('--LOSSWEIGHT', type=float, default=0.,
                    help='comb loss')
parser.add_argument('--ROUND', type=float, default=0.,
                    help='comb loss')
args = parser.parse_args()


def arrtoimg(img):
    img = tf.depth_to_space(tf.reshape(img, [1, args.BSIZE, args.BSIZE, args.SIZE * args.SIZE * 3]), args.SIZE)
    img = tf.reshape(img, [args.SIZE*args.BSIZE, args.SIZE*args.BSIZE, 3])
    img = sess.run(img)
    img = np.clip(img, 0., 1.) * 255
    img = np.uint8(img)
    img = Image.fromarray(img)
    return img


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
rnd = tf.placeholder(shape=[], dtype=tf.float32)
comp = tf.placeholder(shape=[], dtype=tf.float32)
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
    net = mdv7.model(data.batch, std_img, std_denoise, comp=comp)

sess.run(tf.global_variables_initializer())
rnds = [4.5, 5., 5.5, 6., 6.5, 7.]
if saved.latest is not None:
    sys.stdout.write("Restoring from " + saved.latest + "\n")
    sys.stdout.flush()
    net.load(saved.latest, sess)
    saved.clean(last=KEEPLAST)

imgs = batcher.get_batch()
_ = sess.run(data.fetchOp, feed_dict=data.getfeed(imgs))
img_size = args.SIZE*args.BSIZE
table_denoise = np.zeros([std_num, 3])
table_compression = np.zeros([std_num, 3])
for i in range(std_num):
    table_denoise[i][0] = stds[i]*10
    table_compression[i][0] = stds[i] * 10
for i in range(args.ITER):
    _ = sess.run(data.swapOp)
    imgs = batcher.get_batch()
    imgname = imgs[0].split('/')[-1][:-4]
    fdict = data.getfeed(imgs)
    new_comp = Image.new('RGB', (img_size * (std_num + 1), img_size + 10), color=(255, 255, 255))
    new_denoi = Image.new('RGB', (img_size * (std_num + 1), img_size + 10), color=(255, 255, 255))
    d_comp = ImageDraw.Draw(new_comp)
    d_denoi = ImageDraw.Draw(new_denoi)
    for j in range(std_num):
        fdict[std_img] = np.repeat(stds[j], args.BSIZE*args.BSIZE)
        # Run training step & getch for next batch
        if j == 0:
            for k in range(std_num):
                fdict[std_denoise] = np.repeat(stds[k], args.BSIZE*args.BSIZE)
                repre, loss, img_output, img_input, _ = sess.run([net.rep, net.loss, net.out, net.inp, data.fetchOp],
                                                    feed_dict=fdict)

                img_inp = arrtoimg(img_input)
                img_out = arrtoimg(img_output)

                if k == 0:
                    new_comp.paste(img_inp, (0, 0))
                new_comp.paste(img_out, ((k + 1) * img_size, 0))
                d_comp.text(((k+1) * img_size, img_size), str(loss), fill=(0, 0, 0))
                num_non_zero = 0
                for repre_i in repre:
                    num_non_zero = num_non_zero+np.count_nonzero(repre_i)
                d_comp.text(((k+1.5) * img_size, img_size), str(num_non_zero/(img_size*img_size*3.0)), fill=(0, 0, 0))
                table_compression[k][1] = table_compression[k][1] + num_non_zero / (img_size * img_size * 3.0)
                table_compression[k][2] = table_compression[k][2] + loss

        fdict[std_denoise] = np.repeat(stds[j], args.BSIZE * args.BSIZE)
        repre, loss, img_output, img_input, _ = sess.run([net.rep, net.loss, net.out, net.inp, data.fetchOp],
                                                         feed_dict=fdict)

        img_inp = arrtoimg(img_input)
        img_out = arrtoimg(img_output)
        if j == 0:
            new_denoi.paste(img_inp, (0, 0))
        new_denoi.paste(img_out, ((j + 1) * img_size, 0))
        d_denoi.text(((j+1) * img_size, img_size), str(loss), fill=(0, 0, 0))
        num_non_zero = 0
        for repre_i in repre:
            num_non_zero = num_non_zero + np.count_nonzero(repre_i)
        d_denoi.text(((j+1.5) * img_size, img_size), str(num_non_zero / (img_size * img_size * 3.0)), fill=(0, 0, 0))
        table_denoise[j][1] = table_denoise[j][1] + num_non_zero / (img_size * img_size * 3.0)
        table_denoise[j][2] = table_denoise[j][2] + loss

    new_comp.save(args.IMGPATH + args.WTS + imgname + '_compression.jpg')
    new_denoi.save(args.IMGPATH + args.WTS + imgname + '_denoising.jpg')

np.savetxt('rep/modelv'+str(args.MD)+'l'+str(args.LOSSWEIGHT)+'_denoise.csv', table_denoise/10., delimiter=",")
np.savetxt('rep/modelv'+str(args.MD)+'l'+str(args.LOSSWEIGHT)+'_compression.csv', table_compression/10., delimiter=",")
