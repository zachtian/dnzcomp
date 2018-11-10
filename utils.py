
import re
import os
from glob import glob
import numpy as np


# Raw load text file
def load(fname):
    data = []
    for line in open(fname).readlines():
        l = line.strip().split(' ')
        data.append(l[0])
    return data


# Reading in batches
# repeatable random shuffling
# Initialize by calling a filename where each line is a string
# (typically a filename) followed by a numerical label.
class batcher:
    def __init__(self, fname, bsz, niter=0):

        # Load from file
        d = load(fname)
        self.data = d

        # Setup batching
        self.bsz = bsz
        
        self.rand = np.random.RandomState(8)
        idx = self.rand.permutation(len(self.data))
        for i in range(niter*bsz // len(idx)):
            idx = self.rand.permutation(len(idx))

        self.idx = np.int32(idx)
        self.pos = niter*bsz % len(self.idx)

    def get_batch(self):
        if self.pos+self.bsz >= len(self.idx):
            bidx = self.idx[self.pos:]

            idx = self.rand.permutation(len(self.idx))
            self.idx = np.int32(idx)

            self.pos = 0
            if len(bidx) < self.bsz:
                self.pos = self.bsz-len(bidx)
                bidx2 = self.idx[0:self.pos]
                bidx = np.concatenate((bidx,bidx2))
        else:
            bidx = self.idx[self.pos:self.pos+self.bsz]
            self.pos = self.pos+self.bsz

        return [self.data[bidx[i]]
                for i in range(len(bidx))]


# Manage checkpoint files, read off iteration number from filename
# Use clean() to keep latest, and modulo n iters, delete rest
class ckpter:
    def __init__(self, wcard):
        self.wcard = wcard
        self.load()


    def load(self):
        lst = glob(self.wcard)
        if len(lst) > 0:
            lst = [(l, int(re.match('.*/.*_(\d+)', l).group(1)))
                 for l in lst]
            self.lst=sorted(lst,key=lambda x: x[1])

            self.iter = self.lst[-1][1]
            self.latest = self.lst[-1][0]
        else:
            self.lst=[]
            self.iter=0
            self.latest=None

    def clean(self, every=0, last=1):
        self.load()
        old = self.lst[:-last]
        for j in old:
            if every == 0 or j[1] % every != 0:
                os.remove(j[0])
