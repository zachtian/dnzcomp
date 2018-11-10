import tensorflow as tf


class testload:
    def __init__(self, size=32, bsize=10):
        self.names = []
        # Create placeholders
        self.names.append(tf.placeholder(tf.string))

        # Load image
        img = tf.read_file(self.names[0])
        img = tf.image.decode_jpeg(img, channels=3)

        # Crop image to 64
        img = tf.random_crop(img, [size*bsize, size*bsize, 3], seed=1)

        img = tf.space_to_depth(tf.reshape(img, [1, size*bsize, size*bsize, 3]), size)

        batch = tf.reshape(img, [-1, size, size, 3])

        batch = tf.to_float(batch) / 255.0
        # Fetching logic
        nBuf = tf.Variable(tf.zeros([bsize*bsize, size, size, 3], dtype=tf.float32), trainable=False)
        self.batch = tf.Variable(tf.zeros([bsize*bsize, size, size, 3], dtype=tf.float32), trainable=False)

        self.fetchOp = tf.assign(nBuf, batch).op
        self.swapOp = tf.assign(self.batch, nBuf)

    def getfeed(self, imgs):
        dict = {}
        for i in range(len(self.names)):
            dict[self.names[i]] = imgs[i]
        return dict
