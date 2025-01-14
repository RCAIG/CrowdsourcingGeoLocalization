import tensorflow as tf
#import tensorflow.compat.v1 as tf
import numpy as np
from utils import tf_shape


def sample_within_bounds(signal, batch_index, y, x, channel_index):
    '''
    :param signal: tf variable, shape = [batch, height, width, channel]
    :param x: numpy
    :param y: numpy
    :return:
    '''

    index = tf.stack([tf.reshape(batch_index, [-1]), tf.reshape(y, [-1]), tf.reshape(x, [-1]),
                      tf.reshape(channel_index, [-1])], axis=1)
    index=tf.cast(index, dtype=tf.int32)
    signal = tf.cast(signal, dtype=tf.int32)

    result = tf.gather_nd(signal, index)

    batch, height, width, channel = tf.shape(x)

    sample = tf.reshape(result, [batch, height, width, channel])

    return sample


def sample_bilinear(signal, batch_index, ry, rx, channel_index):
    '''
    :param signal: tensor_shape = [batch, sat_height, sat_width, channel]
    :param rx: tensor_shape = [batch, grd_height, grd_width, channel]
    :param ry: tensor_shape = [batch, grd_height, grd_width, channel]
    :param batch_index: tensor_shape = [batch, grd_height, grd_width, channel]
    :param channel_index: tensor_shape = [batch, grd_height, grd_width, channel]
    :return:
    '''

    signal_dim_y, signal_dim_x = signal.get_shape().as_list()[1:-1]

    # obtain four sample coordinates
    ix0 = tf.cast(rx, tf.float32)
    iy0 = tf.cast(ry, tf.float32)
    ix1 = tf.minimum(ix0 + 1, signal_dim_x-1)
    iy1 = tf.minimum(iy0 + 1, signal_dim_y-1)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, batch_index, iy0, ix0, channel_index)
    signal_10 = sample_within_bounds(signal, batch_index, iy0, ix1, channel_index)
    signal_01 = sample_within_bounds(signal, batch_index, iy1, ix0, channel_index)
    signal_11 = sample_within_bounds(signal, batch_index, iy1, ix1, channel_index)

    signal_00 = tf.cast(signal_00, dtype=tf.float32)
    signal_10 = tf.cast(signal_10, dtype=tf.float32)
    signal_01 = tf.cast(signal_01, dtype=tf.float32)
    signal_11 = tf.cast(signal_11, dtype=tf.float32)

    ix1 = tf.cast(ix1, tf.float32)
    iy1 = tf.cast(iy1, tf.float32)
    # linear interpolation in x-direction
    fx1 = (ix1 - rx) * signal_00 + (1 - ix1 + rx) * signal_10
    fx2 = (ix1 - rx) * signal_01 + (1 - ix1 + rx) * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry) * fx1 + (1 - iy1 + ry) * fx2


def polar_transformer(signal, height, width, shift, max_shift=20):
    '''
    :param signal: [batch, size, size, channel]
    :param height: scalar
    :param width:  scalar
    :param shift:  [batch, 2]
    :return:
    '''

    batch, S, _, channel = tf_shape(signal, 4)

    b = tf.range(0, batch)
    h = tf.range(0, height)
    w = tf.range(0, width)
    c = tf.range(0, channel)

    bb, hh, ww, cc = tf.meshgrid(b, h, w, c, indexing='ij')

    hh = tf.cast(hh, tf.float32)
    ww = tf.cast(ww, tf.float32)

    shift = tf.reshape(shift, [batch, 1, 1, 2])

    shift_x = shift[..., 0:1]
    shift_y = shift[..., 1:2]

    shift_x = tf.tile(shift_x, [1, height, width, channel])
    shift_y = tf.tile(shift_y, [1, height, width, channel])

    radius = S/2. - max_shift

    x = (S / 2. + shift_x) - radius / height * (height - 1 - hh) * tf.sin(2 * np.pi * ww / width)
    y = (S / 2. + shift_y) + radius / height * (height - 1 - hh) * tf.cos(2 * np.pi * ww / width)

    return sample_bilinear(signal, bb, y, x, cc)


def geometry_projector(signal, height, width, shift):
    #batch, S, _, channel = tf_shape(signal, 4)
    batch, S, _, channel = tf.shape(signal)
    S = tf.cast(S, dtype=tf.float32)

    b = tf.range(0, batch)
    b=tf.cast(b, dtype=tf.float32)
    h = tf.range(0, height*2)
    h = tf.cast(h, dtype=tf.float32)
    w = tf.range(0, width)
    w = tf.cast(w, dtype=tf.float32)
    c = tf.range(0, channel)
    c = tf.cast(c, dtype=tf.float32)

    bb, hh, ww, cc = tf.meshgrid(b, h, w, c, indexing='ij')

    hh = tf.cast(hh, tf.float32)
    ww = tf.cast(ww, tf.float32)

    shift = tf.reshape(shift, [batch, 1, 1, 2])

    shift_x = shift[..., 0:1]
    shift_y = shift[..., 1:2]

    shift_x = tf.tile(shift_x, [1, height*2, width, channel])
    shift_y = tf.tile(shift_y, [1, height*2, width, channel])

    tanhh = tf.tan(hh * np.pi/(height*2))
    grd_height = -2
    s = S/50.
    x_2=s*grd_height*tanhh*tf.sin(2 * np.pi * ww / width)

    x_bottom = ((S/2. + shift_x) - s*grd_height*tanhh*tf.sin(2 * np.pi * ww / width))[:, height:, :, :]
    y_bottom = ((S/2. + shift_y) + s*grd_height*tanhh*tf.cos(2 * np.pi * ww / width))[:, height:, :, :]

    x_half = -1 * tf.ones([batch, height, width, channel])
    y_half = -1 * tf.ones([batch, height, width, channel])

    x = tf.concat([x_half, x_bottom], axis=1)
    y = tf.concat([y_half, y_bottom], axis=1)

    projected_signal = sample_bilinear(signal, bb, y, x, cc)

    return projected_signal[:, int(height/2): -int(height/2)]


if __name__=='__main__':

    from PIL import Image
    img = Image.open('_1EPO6UO0lF4EWu_lTsNjg_satView.jpg').resize((256, 256))
    img = np.asarray(img).astype(np.float32)
    img = tf.constant(img)
    batch = 2

    signal = tf.stack([img]*batch, axis=0) # shape = [batch, 256, 256, 3]
    shift = np.random.randint(-10, 10, size=(batch, 2)).astype(np.float32)
    print(shift[0, 0])
    print(shift[0, 1])

    shift_tf = tf.constant(shift)

    image = geometry_projector(signal, 128, 512, shift_tf)

    # sess = tf.Session()
    #
    # images = sess.run(image)
    images = image.numpy()

    image0 = Image.fromarray(images[0].astype(np.uint8))
    image0.save('tan0.png')
    image1 = Image.fromarray(images[1].astype(np.uint8))
    image1.save('tan1.png')
