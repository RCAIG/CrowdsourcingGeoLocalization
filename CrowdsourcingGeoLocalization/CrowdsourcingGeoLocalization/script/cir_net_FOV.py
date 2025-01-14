# import tensorflow as tf

from IAFF import iAFF
from AFF import AFF
from convnext_b import convnext_b
# from utils import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.models import Model



def tf_shape(x, rank):
    static_shape = x.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(x), rank)
    return [s if s is not None else d for s,d in zip(static_shape, dynamic_shape)]


def corr(sat_matrix, crd_matrix):

    s_h, s_w, s_c = sat_matrix.get_shape().as_list()[1:]
    g_h, g_w, g_c = crd_matrix.get_shape().as_list()[1:]

    assert s_h == g_h, s_c == g_c

    def warp_pad_columns(x, n):
        out = tf.concat([x, x[:, :, :n, :]], axis=2)
        return out

    n = g_w - 1
    x = warp_pad_columns(sat_matrix, n)
    f = tf.transpose(crd_matrix, [1, 2, 3, 0])
    out = tf.nn.conv2d(x, f,  strides=[1, 1, 1, 1], padding='VALID')
    h, w = out.get_shape().as_list()[1:-1]
    assert h==1, w==s_w

    out = tf.squeeze(out)  # shape = [batch_sat, w, batch_crd]
    orien = tf.argmax(out, axis=1)  # shape = [batch_sat, batch_crd]

    return out, tf.cast(orien, tf.int32)


def crop_sat(sat_matrix, orien, crd_width):
    batch_sat, batch_crd = tf_shape(orien, 2)
    h, w, channel = sat_matrix.get_shape().as_list()[1:]
    sat_matrix = tf.expand_dims(sat_matrix, 1) # shape=[batch_sat, 1, h, w, channel]
    sat_matrix = tf.tile(sat_matrix, [1, batch_crd, 1, 1, 1])
    sat_matrix = tf.transpose(sat_matrix, [0, 1, 3, 2, 4])  # shape = [batch_sat, batch_crd, w, h, channel]

    orien = tf.expand_dims(orien, -1) # shape = [batch_sat, batch_crd, 1]

    i = tf.range(batch_sat)
    j = tf.range(batch_crd)
    k = tf.range(w)
    x, y, z = tf.meshgrid(i, j, k, indexing='ij')

    z_index = tf.mod(z + orien, w)
    x1 = tf.reshape(x, [-1])
    y1 = tf.reshape(y, [-1])
    z1 = tf.reshape(z_index, [-1])
    index = tf.stack([x1, y1, z1], axis=1)

    sat = tf.reshape(tf.gather_nd(sat_matrix, index), [batch_sat, batch_crd, w, h, channel])

    index1 = tf.range(crd_width)
    sat_crop_matrix = tf.transpose(tf.gather(tf.transpose(sat, [2, 0, 1, 3, 4]), index1), [1, 2, 3, 0, 4])
    # shape = [batch_sat, batch_crd, h, crd_width, channel]
    assert sat_crop_matrix.get_shape().as_list()[3] == crd_width

    return sat_crop_matrix

def corr_crop_distance(sat_vgg, crd_vgg):
    corr_out, corr_orien = corr(sat_vgg, crd_vgg)
    sat_cropped = crop_sat(sat_vgg, corr_orien, crd_vgg.get_shape().as_list()[2])
    # shape = [batch_sat, batch_crd, h, crd_width, channel]

    sat_matrix = tf.nn.l2_normalize(sat_cropped, axis=[2, 3, 4])

    distance = 2 - 2 * tf.transpose(tf.reduce_sum(sat_matrix * tf.expand_dims(crd_vgg, axis=0), axis=[2, 3, 4]))
    # shape = [batch_crd, batch_sat]

    return sat_matrix, distance, corr_orien

def ConvNext_conv_three_iaff(x_crd, x_sat, x_grd, keep_prob, trainable):
    conv_crd = convnext_b(x_crd, keep_prob, 16, trainable, 'Conv_crd')
    crd_conv = tf.nn.l2_normalize(conv_crd, axis=[1, 2, 3])  # shape = [batch, 4, 5, 16]

    conv_grd = convnext_b(x_grd, keep_prob, 16, trainable, 'Conv_grd')
    grd_conv = tf.nn.l2_normalize(conv_grd, axis=[1, 2, 3])  # shape = [batch, 4, 16, 8]

    conv_sat = convnext_b(x_sat, keep_prob, 16, trainable, 'Conv_sat')
    sat_conv = tf.nn.l2_normalize(conv_sat, axis=[1, 2, 3])  # shape = [batch, 4, 16, 8]

    channels = grd_conv.shape[-1]
    model = iAFF(channels=channels)
    mul_conv = model(grd_conv, sat_conv)

    mul_matrix, distance, pred_orien = corr_crop_distance(mul_conv, crd_conv)

    return mul_conv, crd_conv, distance, pred_orien
