import tensorflow as tf
import numpy as np
import trainer as tr
# Rate at which batch-norm population averages decay
_bndecay = 0.99


class model:
    # Add Gaussian noise to Image
    def gaussian_noise(self, inp, std):

        noise = tf.random_normal(shape=np.shape(inp), dtype=tf.float32) * std
        return inp + noise

    def conv(self, inp, ksz, name, stride=1, padding="VALID", ifrelu=True, ifbn=True, ifinpbnorm=False, noise_inp=None):
        ksz = [ksz[0], ksz[0], ksz[1], ksz[2]]

        # xavier init
        sq = np.sqrt(3.0 / np.float32(ksz[0] * ksz[1] * ksz[2]))
        w = tf.Variable(tf.random_uniform(ksz, minval=-sq, maxval=sq, dtype=tf.float32))

        self.weights[name + '_W'] = w

        # constant init
        b = tf.Variable(tf.constant(0, shape=[ksz[3]], dtype=tf.float32))
        self.weights[name + '_b'] = b

        if not noise_inp == None:
            b_noise = tf.Variable(tf.constant(0, shape=[ksz[3]], dtype=tf.float32))
            self.weights[name + '_b_noise'] = b_noise

        # Add conv layer with bias
        out = tf.nn.conv2d(inp, w, [1, stride, stride, 1], padding)
        # Batch-normalization
        if ifbn:
            if ifinpbnorm:
                mn = tf.Variable(tf.random_uniform([ksz[3]], dtype=tf.float32))
                vr = tf.Variable(tf.random_uniform([ksz[3]], dtype=tf.float32))
            else:
                mn, vr = tf.nn.moments(out, axes=[0, 1, 2])
            self.bnorm_mn[name + '_mn'] = mn
            self.bnorm_vr[name + '_vr'] = vr
            out = tf.nn.batch_normalization(out, mn, vr, None, None, 1e-6)

        out = out +  b

        if not noise_inp == None:
            out = out + b_noise * noise_inp

        # ReLU
        if ifrelu:
            out = tf.nn.relu(out)

        return out

    def trans_conv(self, inp, ksz, name, stride=1, padding="VALID", ifrelu=True, ifbn=True, ifinpbnorm=False):
        ksz = [ksz[0], ksz[0], ksz[2], ksz[1]]

        # xavier init
        sq = np.sqrt(3.0 / np.float32(ksz[0] * ksz[1] * ksz[3]))
        w = tf.Variable(tf.random_uniform(ksz, minval=-sq, maxval=sq, dtype=tf.float32))
        self.weights[name + '_W'] = w

        # constant init
        b = tf.Variable(tf.constant(0, shape=[ksz[2]], dtype=tf.float32))
        self.weights[name + '_b'] = b
        input_shape = inp.get_shape()
        out_size = int((input_shape[1] - 1) * stride + ksz[0])
        output_shape = [int(input_shape[0]), out_size, out_size, ksz[2]]
        # Add conv layer with bias
        out = tf.nn.conv2d_transpose(inp, w, output_shape=tf.convert_to_tensor(output_shape),
                                     strides=[1, stride, stride, 1], padding=padding)

        # Batch-normalization
        if ifbn:
            if ifinpbnorm:
                mn = tf.Variable(tf.random_uniform([ksz[2]], dtype=tf.float32))
                vr = tf.Variable(tf.random_uniform([ksz[2]], dtype=tf.float32))
            else:
                mn, vr = tf.nn.moments(out, axes=[0, 1, 2])
            self.bnorm_mn[name + '_mn'] = mn
            self.bnorm_vr[name + '_vr'] = vr
            out = tf.nn.batch_normalization(out, mn, vr, None, None, 1e-6)

        out = out + b

        # ReLU
        if ifrelu:
            out = tf.nn.relu(out)

        return out

    def __init__(self, inp, delta_img, delta_denoise, ifbn=False, inp_bnorm=False):
        self.weights = {}
        self.bnorm_mn = {}
        self.bnorm_vr = {}
        self.inp_bnorm = inp_bnorm

        numc = [4, 2, 2]
        numw = [16, 64, 256]
        numd = [4, 2, 1]
        delta_img = tf.reshape(delta_img, [inp.get_shape().as_list()[0], 1, 1, 1])
        delta_denoise = tf.reshape(delta_denoise, [inp.get_shape().as_list()[0], 1, 1, 1])
        self.gts = inp
        out = self.gaussian_noise(inp, delta_img)
        self.inp = out
        out = out - 0.5
        # Representation on different scales
        rep = []

        # conv

        prev = 3
        cur = 16

        for i in range(len(numc)):
            for j in range(numc[i]):
                out = self.conv(out, [3, prev, cur],
                                'conv' + str(i + 1) + '_' + str(j + 1), ifbn=ifbn, ifinpbnorm=inp_bnorm)
                prev = cur

            # Append all the representations
            if i < 2:

                out_rep = self.conv(out, [1, cur, int(cur / numd[i])],
                                    'conv' + str(i + 1) + '_' + str(j + 5), ifbn=ifbn, ifinpbnorm=inp_bnorm, noise_inp=delta_denoise)
                cur = int(cur / numd[i])
                out_rep = self.conv(out_rep, [1, cur, int(cur / numd[i])],
                                    'conv' + str(i + 1) + '_' + str(j + 6), ifbn=False, ifinpbnorm=inp_bnorm, noise_inp=delta_denoise)
                rep.append(out_rep)

                cur = int(numw[i+1])
                out = self.conv(out, [2, prev, cur],
                                'conv' + str(i + 1) + '_' + str(j + 4), stride=2, ifbn=ifbn, ifinpbnorm=inp_bnorm)
                prev = cur
            else:
                out_rep = self.conv(out, [1, cur, cur],
                                    'conv' + str(i + 1) + '_' + str(j + 5), ifbn=ifbn, ifinpbnorm=inp_bnorm, noise_inp=delta_denoise)
                out_rep = self.conv(out_rep, [1, cur, cur],
                                    'conv' + str(i + 1) + '_' + str(j + 6), ifbn=False, ifinpbnorm=inp_bnorm, noise_inp=delta_denoise)
                rep.append(out_rep)


        self.rep = rep
        # Trans_conv
        numc = [3, 2, 2]
        numoutw = [64, 16, 3]
        numinw = [256, 16, 1]
        for i in range(len(numc)):
            j = 0
            cur = numinw[i]
            if i == 0:
                out = self.trans_conv(rep[2-i], [1, cur, cur],
                                           'trans_conv' + str(i + 1) + '_' + str(j + 1), ifbn=ifbn, ifinpbnorm=inp_bnorm)
                out = self.trans_conv(out, [1, cur, cur],
                                      'trans_conv' + str(i + 1) + '_' + str(j + 2), ifbn=ifbn, ifinpbnorm=inp_bnorm)
            else:
                out_temp = self.trans_conv(rep[2-i], [1, cur, cur*numd[2-i]],
                                           'trans_conv' + str(i + 1) + '_' + str(j + 1), ifbn=ifbn, ifinpbnorm=inp_bnorm)
                cur = cur * numd[2 - i]
                out_temp = self.trans_conv(out_temp, [1, cur, cur*numd[2-i]],
                                           'trans_conv' + str(i + 1) + '_' + str(j + 2), ifbn=ifbn,
                                           ifinpbnorm=inp_bnorm)
                cur = cur * numd[2 - i]
                out = tf.concat([out, out_temp], 3)
                out = self.trans_conv(out, [1, cur*2, cur],
                                      'trans_conv' + str(i + 1) + '_' + str(j + 3), ifbn=ifbn, ifinpbnorm=inp_bnorm)

            prev = numw[2-i]
            cur = numw[2-i]
            for j in range(numc[2-i]):
                out = self.trans_conv(out, [3, prev, cur],
                                      'trans_conv' + str(i + 1) + '_' + str(j + 4), ifbn=ifbn, ifinpbnorm=inp_bnorm)
                prev = cur
            cur = numoutw[i]
            if i < 2:
                out = self.trans_conv(out, [2, prev, cur],
                                      'trans_conv' + str(i + 1) + '_' + str(j + 7), stride=2, ifbn=ifbn, ifinpbnorm=inp_bnorm)
            else:
                out = self.trans_conv(out, [3, prev, cur],
                                      'trans_conv' + str(i + 1) + '_' + str(j + 7), ifbn=False)

        self.out = out
        self.loss_out = tf.reduce_mean(tf.square(self.out - self.gts), axis=[1, 2, 3])
        self.loss = tf.reduce_mean(self.loss_out)

    # Load weights from an npz file
    def load(self, fname, sess):
        wts = np.load(fname)
        for k in wts.keys():
            wvar = self.weights[k]
            wk = wts[k].reshape(wvar.get_shape())
            sess.run(wvar.assign(wk))

    def load_bnorm_mn(self, fname, sess):
        mn = np.load(fname)
        for k in mn.keys():
            wvar = self.bnorm_mn[k]
            wk = mn[k].reshape(wvar.get_shape())
            sess.run(wvar.assign(wk))

    def load_bnorm_vr(self, fname, sess):
        vr = np.load(fname)
        for k in vr.keys():
            wvar = self.bnorm_vr[k]
            wk = vr[k].reshape(wvar.get_shape())
            sess.run(wvar.assign(wk))

    # Save weights to an npz file
    def save(self, fname, sess):
        wts = {}
        for k in self.weights.keys():
            wts[k] = self.weights[k].eval(sess)
        np.savez(fname, **wts)

    def saveAdam(self, fname, opt, sess):
        weights = {}
        beta1_power, beta2_power = opt._get_beta_accumulators()
        weights['b1p'] = beta1_power.eval(sess)
        weights['b2p'] = beta2_power.eval(sess)
        for nm in self.weights.keys():
            v = self.weights[nm]
            weights['m_%s' % nm] = opt.get_slot(v, 'm').eval(sess)
            weights['v_%s' % nm] = opt.get_slot(v, 'v').eval(sess)
        np.savez(fname, **weights)

    def loadAdam(self, fname, opt, sess):
        weights = np.load(fname)
        beta1_power, beta2_power = opt._get_beta_accumulators()
        beta1_power.load(weights['b1p'], sess)
        beta2_power.load(weights['b2p'], sess)

        for nm in self.weights.keys():
            v = self.weights[nm]
            opt.get_slot(v, 'm').load(weights['m_%s' % nm], sess)
            opt.get_slot(v, 'v').load(weights['v_%s' % nm], sess)