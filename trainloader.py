import tensorflow as tf


class trainload:
    def __init__(self, bsz=176, size=32):
        self.names = []
        # Create placeholders
        for i in range(11):
            self.names.append(tf.placeholder(tf.string))

        for i in range(11):
            # Load image
            img = tf.read_file(self.names[i])
            img = tf.image.decode_jpeg(img, channels=3)

            # Crop image to 64
            img = tf.random_crop(img, [size*4, size*4, 3])

            img = tf.image.random_flip_left_right(img)

            img = tf.space_to_depth(tf.reshape(img, [1, size*4, size*4, 3]), size)

            if i > 0:
                batch = tf.concat([batch, tf.reshape(img, [-1, size, size, 3])], 0)
            else:
                batch = tf.reshape(img, [-1, size, size, 3])

        batch = tf.to_float(batch) / 255.0

        # Fetching logic
        nBuf = tf.Variable(tf.zeros([bsz, size, size, 3], dtype=tf.float32), trainable=False)
        self.batch = tf.Variable(tf.zeros([bsz, size, size, 3], dtype=tf.float32), trainable=False)

        self.fetchOp = tf.assign(nBuf, batch).op
        self.swapOp = tf.assign(self.batch, nBuf)

    def getfeed(self, imgs):
        dict = {}
        for i in range(len(self.names)):
            dict[self.names[i]] = imgs[i]
        return dict
