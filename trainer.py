import tensorflow as tf
import numpy as np

class train:
    def __init__(self, model, gts, lr, wd, lossweight):
        # L2 loss with the ground truth and the output
        self.loss_out = tf.reduce_mean(tf.square(model.out-gts), axis=[1, 2, 3])
        self.loss = tf.reduce_mean(self.loss_out)

        if wd > 0.:
            # Define L2 weight-decay on all non-bias vars
            reg = list()
            for k in model.weights.keys():
                wt = model.weights[k]
                if len(wt.get_shape()) > 1:
                    reg.append(tf.nn.l2_loss(wt))
                    self.reg = tf.add_n(reg)

                    # This is our minimization objective
                    self.obj = self.loss + wd*self.reg
        else:
            self.obj = self.loss

        # Set up adam trainer

        self.optimizer = tf.train.AdamOptimizer(lr)
        self.opt = self.optimizer.minimize(self.loss)

        if lossweight > 0.:
            self.sparsity = 0
            for i in range(len(model.rep)):
                self.sparsity = self.sparsity + tf.reduce_mean(model.rep[i])
            self.comb_loss = self.loss + lossweight * self.sparsity
            self.opt = self.optimizer.minimize(self.comb_loss)
