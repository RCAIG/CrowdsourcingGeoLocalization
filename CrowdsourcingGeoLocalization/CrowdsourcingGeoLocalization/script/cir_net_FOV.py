# import tensorflow as tf

from VGG import VGG16
from Alexnet import AlexNet
from resnet import resnet101, resnet50,resnet34, resnet18
from VGG_cir import VGG16_cir
from mobilenet import MobileNet
from densenet import densenet
from shufflenet import shufflenet
from InceptionResNetV2 import InceptionResNetV2
from InceptionV3 import InceptionV3
from IAFF import iAFF
from AFF import AFF
from SqueezeNet import squeezenet
from EfficientNet import EfficientNet
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




def VGG_13_conv_crd_sat_cir(x_sat, x_crd, keep_prob, trainable):
    vgg_crd = VGG16(x_crd, keep_prob, trainable, 'VGG_crd')
    crd_layer13 = vgg_crd.layer13_output
    crd_vgg = vgg_crd.conv2(crd_layer13, 'crd', dimension=16)# shape = [batch, 4, 64, 16]
    #crd_vgg2 = vgg_crd.conv2(crd_layer13, 'crd2', dimension=8)
    crd_vgg = tf.nn.l2_normalize(crd_vgg, axis=[1, 2, 3])
    #crd_vgg2 = tf.nn.l2_normalize(crd_vgg2, axis=[1, 2, 3])

    #vgg_sat = VGG16_cir(x_sat, keep_prob, trainable, 'VGG_sat')
    vgg_sat = VGG16(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_layer13 = vgg_sat.layer13_output

    #sat_vgg = vgg_sat.conv2(sat_layer13, 'sat', dimension=16)
    #sat_vgg = tf.nn.l2_normalize(sat_vgg, axis=[1, 2, 3])

    sat_vgg = vgg_sat.conv2(sat_layer13, 'sat', dimension=16)# shape = [batch, 4, 64, 16]
    sat_vgg = tf.nn.l2_normalize(sat_vgg, axis=[1, 2, 3])
    #sat_vgg = tf.concat([crd_vgg2, sat_vgg],axis=-1)


    sat_matrix, distance, pred_orien = corr_crop_distance(sat_vgg, crd_vgg)

    return sat_vgg, crd_vgg, distance, pred_orien

def Alexnet_conv_crd_sat_cir(x_sat, x_crd, keep_prob, trainable):
    alex_crd = AlexNet(x_crd, keep_prob, trainable, 'Alex_crd')
    crd_layer5_output = alex_crd.layer5_output
    crd_alex = alex_crd.conv2(crd_layer5_output, 'crd', dimension=16)
    crd_alex = tf.nn.l2_normalize(crd_alex, axis=[1, 2, 3])  # shape = [batch, 32, 42, 16]

    alex_sat = AlexNet(x_sat, keep_prob, trainable, 'Alex_sat')
    sat_layer5_output = alex_sat.layer5_output
    sat_alex = alex_sat.conv2(sat_layer5_output, 'sat', dimension=16)
    sat_alex = tf.nn.l2_normalize(sat_alex, axis=[1, 2, 3])  # shape = [batch, 32, 42, 16]


    sat_matrix, distance, pred_orien = corr_crop_distance(sat_alex, crd_alex)

    return sat_alex, crd_alex, distance, pred_orien

def VGG_13_conv_crd_grd_cir(x_grd, x_crd, keep_prob, trainable):
    vgg_crd = VGG16(x_crd, keep_prob, trainable, 'VGG_crd')
    crd_layer13 = vgg_crd.layer13_output
    crd_vgg = vgg_crd.conv2(crd_layer13, 'crd', dimension=16)# shape = [batch, 4, 64, 16]
    crd_vgg = tf.nn.l2_normalize(crd_vgg, axis=[1, 2, 3])

    vgg_grd = VGG16(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_layer13 = vgg_grd.layer13_output
    grd_vgg = vgg_grd.conv2(grd_layer13, 'grd', dimension=16)# shape = [batch, 4, 64, 16]
    grd_vgg = tf.nn.l2_normalize(grd_vgg, axis=[1, 2, 3])


    grd_matrix, distance, pred_orien = corr_crop_distance(grd_vgg, crd_vgg)

    return grd_vgg, crd_vgg, distance, pred_orien

def Alexnet_conv_crd_grd_cir(x_grd, x_crd, keep_prob, trainable):
    alex_crd = AlexNet(x_crd, keep_prob, trainable, 'Alex_crd')
    crd_layer5_output = alex_crd.layer5_output
    crd_alex = alex_crd.conv2(crd_layer5_output, 'crd', dimension=16)
    crd_alex = tf.nn.l2_normalize(crd_alex, axis=[1, 2, 3])# shape = [batch, 32, 42, 16]

    alex_grd = AlexNet(x_grd, keep_prob, trainable, 'Alex_grd')
    grd_layer5_output = alex_grd.layer5_output
    grd_alex = alex_grd.conv2(grd_layer5_output, 'grd', dimension=16)
    grd_alex = tf.nn.l2_normalize(grd_alex, axis=[1, 2, 3])  # shape = [batch, 32, 42, 16]

    grd_matrix, distance, pred_orien = corr_crop_distance(grd_alex, crd_alex)

    return grd_alex, crd_alex, distance, pred_orien

def Resnet18_conv_crd_grd_cir(x_grd, x_crd, keep_prob, trainable):
    res_crd = resnet18(x_crd, keep_prob, 16, trainable, 'Res_crd')
    crd_res = tf.nn.l2_normalize(res_crd, axis=[1, 2, 3])  # shape = [batch, 32, 42, 16]

    res_grd = resnet18(x_grd, keep_prob, 16, trainable, 'Res_grd')
    grd_res = tf.nn.l2_normalize(res_grd, axis=[1, 2, 3])  # shape = [batch, 32, 42, 16]


    grd_matrix, distance, pred_orien = corr_crop_distance(grd_res, crd_res)

    return grd_res, crd_res, distance, pred_orien

def Resnet34_conv_crd_grd_cir(x_grd, x_crd, keep_prob, trainable):
    res_crd = resnet34(x_crd, keep_prob, 16, trainable, 'Res_crd')
    crd_res = tf.nn.l2_normalize(res_crd, axis=[1, 2, 3])  # shape = [batch, 32, 42, 16]

    res_grd = resnet34(x_grd, keep_prob, 16, trainable, 'Res_grd')
    grd_res = tf.nn.l2_normalize(res_grd, axis=[1, 2, 3])  # shape = [batch, 32, 42, 16]


    grd_matrix, distance, pred_orien = corr_crop_distance(grd_res, crd_res)

    return grd_res, crd_res, distance, pred_orien

def densenet_conv_crd_grd_cir(x_grd, x_crd, keep_prob, trainable):
    res_crd = densenet(x_crd, keep_prob, 16, trainable, 'Res_crd')
    crd_res = tf.nn.l2_normalize(res_crd, axis=[1, 2, 3])  # shape = [batch, 32, 42, 16]

    res_grd = densenet(x_grd, keep_prob, 16, trainable, 'Res_grd')
    grd_res = tf.nn.l2_normalize(res_grd, axis=[1, 2, 3])  # shape = [batch, 32, 42, 16]


    grd_matrix, distance, pred_orien = corr_crop_distance(grd_res, crd_res)

    return grd_res, crd_res, distance, pred_orien

def VGG_13_conv_three_cir(x_crd, x_sat, x_grd, keep_prob, trainable):
    vgg_crd = VGG16(x_crd, keep_prob, trainable, 'VGG_crd')
    crd_layer13 = vgg_crd.layer13_output
    crd_vgg = vgg_crd.conv2(crd_layer13, 'crd', dimension=16)
    crd_vgg = tf.nn.l2_normalize(crd_vgg, axis=[1, 2, 3])# shape = [batch, 4, 64, 16]

    vgg_grd = VGG16(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_layer13 = vgg_grd.layer13_output
    grd_vgg = vgg_grd.conv2(grd_layer13, 'grd', dimension=8)# shape = [batch, 4, 64, 8]
    grd_vgg = tf.nn.l2_normalize(grd_vgg, axis=[1, 2, 3])

    #vgg_sat = VGG16_cir(x_sat, keep_prob, trainable, 'VGG_sat')
    vgg_sat = VGG16(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_layer13 = vgg_sat.layer13_output

    #sat_vgg = vgg_sat.conv2(sat_layer13, 'sat', dimension=16)
    #sat_vgg = tf.nn.l2_normalize(sat_vgg, axis=[1, 2, 3])

    sat_vgg = vgg_sat.conv2(sat_layer13, 'sat', dimension=8)# shape = [batch, 4, 64, 8]

    mul_vgg = tf.concat([grd_vgg, sat_vgg],axis=-1)


    mul_matrix, distance, pred_orien = corr_crop_distance(mul_vgg, crd_vgg)

    return mul_vgg, crd_vgg, distance, pred_orien

def VGG_13_conv_three_cir_merged(x_crd, x_sat, x_grd, keep_prob, trainable):
    vgg_crd = VGG16(x_crd, keep_prob, trainable, 'VGG_crd')
    crd_layer13 = vgg_crd.layer13_output
    crd_vgg = vgg_crd.conv2(crd_layer13, 'crd', dimension=16)
    crd_vgg = tf.nn.l2_normalize(crd_vgg, axis=[1, 2, 3])# shape = [batch, 4, 64, 16]

    vgg_grd = VGG16(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_layer13 = vgg_grd.layer13_output
    grd_vgg = vgg_grd.conv2(grd_layer13, 'grd', dimension=8)# shape = [batch, 4, 64, 8]
    grd_vgg = tf.nn.l2_normalize(grd_vgg, axis=[1, 2, 3])

    #vgg_sat = VGG16_cir(x_sat, keep_prob, trainable, 'VGG_sat')
    vgg_sat = VGG16(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_layer13 = vgg_sat.layer13_output

    #sat_vgg = vgg_sat.conv2(sat_layer13, 'sat', dimension=16)
    #sat_vgg = tf.nn.l2_normalize(sat_vgg, axis=[1, 2, 3])

    sat_vgg = vgg_sat.conv2(sat_layer13, 'sat', dimension=8)# shape = [batch, 4, 64, 8]

    merged = tf.concat([grd_vgg, sat_vgg], axis=-1)
    mul_vgg = tf.layers.dense(merged, units=16, activation=tf.nn.relu)

    mul_matrix, distance, pred_orien = corr_crop_distance(mul_vgg, crd_vgg)

    return mul_vgg, crd_vgg, distance, pred_orien

def VGG_13_conv_three_cir_iaff(x_crd, x_sat, x_grd, keep_prob, trainable):
    vgg_crd = VGG16(x_crd, keep_prob, trainable, 'VGG_crd')
    crd_layer13 = vgg_crd.layer13_output
    crd_vgg = vgg_crd.conv2(crd_layer13, 'crd', dimension=16)
    crd_vgg = tf.nn.l2_normalize(crd_vgg, axis=[1, 2, 3])# shape = [batch, 4, 64, 16]

    vgg_grd = VGG16(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_layer13 = vgg_grd.layer13_output
    grd_vgg = vgg_grd.conv2(grd_layer13, 'grd', dimension=16)# shape = [batch, 4, 64, 8]
    grd_vgg = tf.nn.l2_normalize(grd_vgg, axis=[1, 2, 3])

    vgg_sat = VGG16(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_layer13 = vgg_sat.layer13_output
    sat_vgg = vgg_sat.conv2(sat_layer13, 'sat', dimension=16)# shape = [batch, 4, 64, 8]

    channels = grd_vgg.shape[-1]
    model = iAFF(channels=channels)
    mul_vgg = model(grd_vgg, sat_vgg)

    mul_matrix, distance, pred_orien = corr_crop_distance(mul_vgg, crd_vgg)

    return mul_vgg, crd_vgg, distance, pred_orien



def VGG_13_conv_three_cir_aff(x_crd, x_sat, x_grd, keep_prob, trainable):
    vgg_crd = VGG16(x_crd, keep_prob, trainable, 'VGG_crd')
    crd_layer13 = vgg_crd.layer13_output
    crd_vgg = vgg_crd.conv2(crd_layer13, 'crd', dimension=16)
    crd_vgg = tf.nn.l2_normalize(crd_vgg, axis=[1, 2, 3])# shape = [batch, 4, 64, 16]

    vgg_grd = VGG16(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_layer13 = vgg_grd.layer13_output
    grd_vgg = vgg_grd.conv2(grd_layer13, 'grd', dimension=16)# shape = [batch, 4, 64, 8]
    grd_vgg = tf.nn.l2_normalize(grd_vgg, axis=[1, 2, 3])

    vgg_sat = VGG16(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_layer13 = vgg_sat.layer13_output
    sat_vgg = vgg_sat.conv2(sat_layer13, 'sat', dimension=16)# shape = [batch, 4, 64, 8]
    channels = grd_vgg.shape[-1]

    model = AFF(channels=channels)
    mul_vgg = model(grd_vgg, sat_vgg)

    mul_matrix, distance, pred_orien = corr_crop_distance(mul_vgg, crd_vgg)

    return mul_vgg, crd_vgg, distance, pred_orien


def VGG_13_conv_three_cir_attention(x_crd, x_sat, x_grd, keep_prob, trainable):
    vgg_crd = VGG16(x_crd, keep_prob, trainable, 'VGG_crd')
    crd_layer13 = vgg_crd.layer13_output
    crd_vgg = vgg_crd.conv2(crd_layer13, 'crd', dimension=16)
    crd_vgg = tf.nn.l2_normalize(crd_vgg, axis=[1, 2, 3])# shape = [batch, 4, 64, 16]

    vgg_grd = VGG16(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_layer13 = vgg_grd.layer13_output
    grd_vgg = vgg_grd.conv2(grd_layer13, 'grd', dimension=8)# shape = [batch, 4, 64, 8]
    grd_vgg = tf.nn.l2_normalize(grd_vgg, axis=[1, 2, 3])

    #vgg_sat = VGG16_cir(x_sat, keep_prob, trainable, 'VGG_sat')
    vgg_sat = VGG16(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_layer13 = vgg_sat.layer13_output

    #sat_vgg = vgg_sat.conv2(sat_layer13, 'sat', dimension=16)
    #sat_vgg = tf.nn.l2_normalize(sat_vgg, axis=[1, 2, 3])

    sat_vgg = vgg_sat.conv2(sat_layer13, 'sat', dimension=8)# shape = [batch, 4, 64, 8]

    # 注意力权重计算
    attention_weights = tf.keras.layers.Dense(1, activation='softmax')(grd_vgg)
    # 将注意力权重应用到第二个特征上
    weighted_sat_vgg = tf.math.multiply(sat_vgg, attention_weights)
    # 注意力融合后的特征
    mul_vgg = tf.concat([grd_vgg, weighted_sat_vgg], axis=-1)  # 大小为 [batch, 4, 64, 16]

    mul_matrix, distance, pred_orien = corr_crop_distance(mul_vgg, crd_vgg)

    return mul_vgg, crd_vgg, distance, pred_orien

def Alexnet_conv_three_cir(x_crd, x_sat, x_grd, keep_prob, trainable):
    alex_crd = AlexNet(x_crd, keep_prob, trainable, 'Alex_crd')
    crd_layer5_output = alex_crd.layer5_output
    crd_alex = alex_crd.conv2(crd_layer5_output, 'crd', dimension=16)
    crd_alex = tf.nn.l2_normalize(crd_alex, axis=[1, 2, 3])  # shape = [batch, 32, 42, 16]

    alex_grd = AlexNet(x_grd, keep_prob, trainable, 'Alex_grd')
    grd_layer5_output = alex_grd.layer5_output
    grd_alex = alex_grd.conv2(grd_layer5_output, 'grd', dimension=8)
    grd_alex = tf.nn.l2_normalize(grd_alex, axis=[1, 2, 3])  # shape = [batch, 32, 42, 8]

    # vgg_sat = VGG16_cir(x_sat, keep_prob, trainable, 'VGG_sat')
    alex_sat = AlexNet(x_sat, keep_prob, trainable, 'Alex_sat')
    sat_layer5_output = alex_sat.layer5_output
    sat_alex = alex_sat.conv2(sat_layer5_output, 'sat', dimension=8)
    sat_alex = tf.nn.l2_normalize(sat_alex, axis=[1, 2, 3])  # shape = [batch, 32, 42, 8]

    mul_alex = tf.concat([grd_alex, sat_alex], axis=-1)

    mul_matrix, distance, pred_orien = corr_crop_distance(mul_alex, crd_alex)

    return mul_alex, crd_alex, distance, pred_orien


def Resnet18_conv_three_cir(x_crd, x_sat, x_grd, keep_prob, trainable):
    res_crd = resnet18(x_crd, keep_prob, 16, trainable, 'Res_crd')
    crd_res = tf.nn.l2_normalize(res_crd, axis=[1, 2, 3])  # shape = [batch, 4, 6, 16]

    res_grd = resnet18(x_grd, keep_prob, 8, trainable, 'Res_grd')
    grd_res = tf.nn.l2_normalize(res_grd, axis=[1, 2, 3])  # shape = [batch, 4, 16, 8]

    # vgg_sat = VGG16_cir(x_sat, keep_prob, 8, ,trainable, 'VGG_sat')
    res_sat = resnet18(x_sat, keep_prob, 8, trainable, 'Res_sat')
    sat_res = tf.nn.l2_normalize(res_sat, axis=[1, 2, 3])  # shape = [batch, 4, 16, 8]

    mul_res = tf.concat([grd_res, sat_res], axis=-1)

    mul_matrix, distance, pred_orien = corr_crop_distance(mul_res, crd_res)

    return mul_res, crd_res, distance, pred_orien

def Resnet34_conv_three_cir(x_crd, x_sat, x_grd, keep_prob, trainable):
    res_crd = resnet34(x_crd, keep_prob, 16, trainable, 'Res_crd')
    crd_res = tf.nn.l2_normalize(res_crd, axis=[1, 2, 3])  # shape = [batch, 4, 6, 16]

    res_grd = resnet34(x_grd, keep_prob, 8, trainable, 'Res_grd')
    grd_res = tf.nn.l2_normalize(res_grd, axis=[1, 2, 3])  # shape = [batch, 4, 16, 8]

    # vgg_sat = VGG16_cir(x_sat, keep_prob, 8, ,trainable, 'VGG_sat')
    res_sat = resnet34(x_sat, keep_prob, 8, trainable, 'Res_sat')
    sat_res = tf.nn.l2_normalize(res_sat, axis=[1, 2, 3])  # shape = [batch, 4, 16, 8]

    mul_res = tf.concat([grd_res, sat_res], axis=-1)

    mul_matrix, distance, pred_orien = corr_crop_distance(mul_res, crd_res)

    return mul_res, crd_res, distance, pred_orien


def mobilenet_conv_three_cir(x_crd, x_sat, x_grd, keep_prob, trainable):
    mobile_crd = MobileNet(x_crd, keep_prob, 16, 1.0,1,1e-3,trainable, 'mobile_crd')
    crd_mobile = tf.nn.l2_normalize(mobile_crd, axis=[1, 2, 3])  # shape = [batch, 4, 6, 16]

    mobile_grd = MobileNet(x_grd, keep_prob, 8, 1.0,1,1e-3,trainable, 'mobile_grd')
    grd_mobile = tf.nn.l2_normalize(mobile_grd, axis=[1, 2, 3])  # shape = [batch, 4, 16, 8]

    # vgg_sat = VGG16_cir(x_sat, keep_prob, 8, ,trainable, 'VGG_sat')
    mobile_sat = MobileNet(x_sat, keep_prob, 8, 1.0,1,1e-3,trainable, 'mobile_sat')
    sat_mobile = tf.nn.l2_normalize(mobile_sat, axis=[1, 2, 3])  # shape = [batch, 4, 16, 8]

    mul_mobile = tf.concat([grd_mobile, sat_mobile], axis=-1)

    mul_matrix, distance, pred_orien = corr_crop_distance(mul_mobile, crd_mobile)

    return mul_mobile, crd_mobile, distance, pred_orien

def Shufflenet_conv_three_cir(x_crd, x_sat, x_grd, keep_prob, trainable):
    shu_crd = shufflenet(x_crd, keep_prob, 16, trainable, 'shu_crd')
    crd_shu = tf.nn.l2_normalize(shu_crd, axis=[1, 2, 3])  # shape = [batch, 4, 6, 16]

    shu_grd = shufflenet(x_grd, keep_prob, 8, trainable, 'shu_grd')
    grd_shu = tf.nn.l2_normalize(shu_grd, axis=[1, 2, 3])  # shape = [batch, 4, 16, 8]

    # vgg_sat = VGG16_cir(x_sat, keep_prob, 8, ,trainable, 'VGG_sat')
    shu_sat = shufflenet(x_sat, keep_prob, 8, trainable, 'shu_sat')
    sat_shu = tf.nn.l2_normalize(shu_sat, axis=[1, 2, 3])  # shape = [batch, 4, 16, 8]

    mul_shu = tf.concat([grd_shu, sat_shu], axis=-1)

    mul_matrix, distance, pred_orien = corr_crop_distance(mul_shu, crd_shu)

    return mul_shu, crd_shu, distance, pred_orien


def InceptionResNetV2_conv_three_cir(x_crd, x_sat, x_grd, keep_prob, trainable):
    inc_crd = InceptionResNetV2(x_crd, keep_prob, 16, trainable, 'inc_crd')
    crd_inc = tf.nn.l2_normalize(inc_crd, axis=[1, 2, 3])  # shape = [batch, 2, 3, 16]

    inc_grd = InceptionResNetV2(x_grd, keep_prob, 8, trainable, 'inc_grd')
    grd_inc = tf.nn.l2_normalize(inc_grd, axis=[1, 2, 3])  # shape = [batch, 2, 14, 8]

    inc_sat = InceptionResNetV2(x_sat, keep_prob, 8, trainable, 'inc_sat')
    sat_inc = tf.nn.l2_normalize(inc_sat, axis=[1, 2, 3])  # shape = [batch, 2, 14, 8]

    mul_inc = tf.concat([grd_inc, sat_inc], axis=-1)

    mul_matrix, distance, pred_orien = corr_crop_distance(mul_inc, crd_inc)

    return mul_inc, crd_inc, distance, pred_orien

def InceptionV3_conv_three_cir(x_crd, x_sat, x_grd, keep_prob, trainable):
    inc_crd = InceptionV3(x_crd, keep_prob, 16, trainable, 'inc_crd')
    crd_inc = tf.nn.l2_normalize(inc_crd, axis=[1, 2, 3])  # shape = [batch, 2, 3, 16]

    inc_grd = InceptionV3(x_grd, keep_prob, 8, trainable, 'inc_grd')
    grd_inc = tf.nn.l2_normalize(inc_grd, axis=[1, 2, 3])  # shape = [batch, 2, 14, 8]

    inc_sat = InceptionV3(x_sat, keep_prob, 8, trainable, 'inc_sat')
    sat_inc = tf.nn.l2_normalize(inc_sat, axis=[1, 2, 3])  # shape = [batch, 2, 14, 8]

    mul_inc = tf.concat([grd_inc, sat_inc], axis=-1)

    mul_matrix, distance, pred_orien = corr_crop_distance(mul_inc, crd_inc)

    return mul_inc, crd_inc, distance, pred_orien
# def InceptionV3_conv_three_cir(x_crd, x_sat, x_grd, keep_prob, trainable):
#     base_model = InceptionV3(weights='imagenet', include_top=False)
#     base_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
#     inc_crd = base_model.predict(x_crd,steps=4)
#     crd_inc = tf.nn.l2_normalize(inc_crd, axis=[1, 2, 3])  # shape = [batch, 2, 3, 16]
#
#     inc_grd = base_model.predict(x_grd,steps=4)
#     grd_inc = tf.nn.l2_normalize(inc_grd, axis=[1, 2, 3])  # shape = [batch, 2, 14, 8]
#
#     inc_sat = base_model.predict(x_sat,steps=4)
#     sat_inc = tf.nn.l2_normalize(inc_sat, axis=[1, 2, 3])  # shape = [batch, 2, 14, 8]
#
#     mul_inc = tf.concat([grd_inc, sat_inc], axis=-1)
#
#     mul_matrix, distance, pred_orien = corr_crop_distance(mul_inc, crd_inc)
#
#     return mul_inc, crd_inc, distance, pred_orien


def squeezenet_conv_three_cir(x_crd, x_sat, x_grd, keep_prob, trainable):
    inc_crd = squeezenet(x_crd, keep_prob, 16, trainable, 'inc_crd')
    crd_inc = tf.nn.l2_normalize(inc_crd, axis=[1, 2, 3])  # shape = [batch, 8, 11, 16]

    inc_grd = squeezenet(x_grd, keep_prob, 8, trainable, 'inc_grd')
    grd_inc = tf.nn.l2_normalize(inc_grd, axis=[1, 2, 3])  # shape = [batch, 8, 32, 8]

    inc_sat = squeezenet(x_sat, keep_prob, 8, trainable, 'inc_sat')
    sat_inc = tf.nn.l2_normalize(inc_sat, axis=[1, 2, 3])  # shape = [batch, 8, 32, 8]

    mul_inc = tf.concat([grd_inc, sat_inc], axis=-1)

    mul_matrix, distance, pred_orien = corr_crop_distance(mul_inc, crd_inc)

    return mul_inc, crd_inc, distance, pred_orien

def EfficientNet_conv_three_cir(x_crd, x_sat, x_grd, keep_prob, trainable):
    model_crd = EfficientNet(width_coefficient=2.0, depth_coefficient=1.1, dropout_rate=0.5, num_classes=16)
    inc_crd = model_crd(x_crd)
    crd_inc = tf.nn.l2_normalize(inc_crd, axis=[1, 2, 3])  # shape = [batch, 4, 6, 16]

    model_grd = EfficientNet(width_coefficient=2.0, depth_coefficient=1.1, dropout_rate=0.5, num_classes=8)
    inc_grd = model_grd(x_grd)
    grd_inc = tf.nn.l2_normalize(inc_grd, axis=[1, 2, 3])  # shape = [batch, 4, 16, 8]

    model_sat = EfficientNet(width_coefficient=2.0, depth_coefficient=1.1, dropout_rate=0.5, num_classes=8)
    inc_sat = model_sat(x_sat)
    sat_inc = tf.nn.l2_normalize(inc_sat, axis=[1, 2, 3])  # shape = [batch, 4, 16, 8]

    mul_inc = tf.concat([grd_inc, sat_inc], axis=-1)

    mul_matrix, distance, pred_orien = corr_crop_distance(mul_inc, crd_inc)

    return mul_inc, crd_inc, distance, pred_orien


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
