import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from cir_net_FOV import *
from distance import *
from OriNet_MSHKED.input_data_hk_polar_crd_grd import InputData
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import argparse
from tensorflow.python.ops.gen_math_ops import *

parser = argparse.ArgumentParser(description='TensorFlow implementation.')


parser.add_argument('--network_type',              type=str,   help='network type',      default='ConvNext_conv_three_iaff')

parser.add_argument('--start_epoch',               type=int,   help='from epoch', default=0)
parser.add_argument('--number_of_epoch',           type=int,   help='number_of_epoch', default=300)
parser.add_argument('--polar',                     type=int,   help='0 or 1',   default=1)

# parser.add_argument('--train_crd_noise',           type=int,   help='0~360',    default=360)
# parser.add_argument('--test_crd_noise',            type=int,   help='0~360',    default=0)
#
# parser.add_argument('--train_crd_FOV',             type=int,   help='70, 90, 100, 120, 180, 360',   default=360)
# parser.add_argument('--test_crd_FOV',              type=int,   help='70, 90, 100, 120, 180, 360',   default=360)

args = parser.parse_args()

# --------------  configuration parameters  -------------- #
network_type = args.network_type

start_epoch = args.start_epoch
polar = args.polar

number_of_epoch = args.number_of_epoch

data_type = 'MSHKED'

loss_type = 'l1'

batch_size = 32
is_training = True
loss_weight = 10.0
# number_of_epoch = 100

mat_type = 'matdata_HK_full_0.2.mat'
learning_rate_val = 1e-5
keep_prob_val = 0.8

dimension = 4
# -------------------------------------------------------- #


def validate(dist_array, topK):
    accuracy = 0.0
    data_amount = 0.0

    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[i, :] < gt_dist)
        if prediction < topK:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy



def compute_loss(dist_array, temperature=0.07):


    with tf.name_scope('info_nce_loss'):
        similarity_matrix = -dist_array / temperature
        positive_pairs = tf.diag_part(similarity_matrix)
        exp_sim_matrix = tf.exp(similarity_matrix)
        mask = tf.ones_like(exp_sim_matrix) - tf.eye(tf.shape(exp_sim_matrix)[0])
        exp_sim_matrix = exp_sim_matrix * mask
        denominator = tf.reduce_sum(exp_sim_matrix, axis=1)
        positive_exp = tf.exp(positive_pairs)
        loss_g2s = -tf.reduce_mean(tf.log(positive_exp / (denominator + 1e-8)))
        loss_s2g = -tf.reduce_mean(tf.log(positive_exp / (tf.reduce_sum(exp_sim_matrix, axis=0) + 1e-8)))
        loss = (loss_g2s + loss_s2g) / 2.0

    return loss

def train(start_epoch=0):

    # import data
    input_data = InputData(polar)

    crd_x = tf.placeholder(tf.float32, [None, 128, 170, 3], name='crd_x')
    grd_x = tf.placeholder(tf.float32, [None, 128, 512, 3], name='grd_x')

    utms_x = tf.placeholder(tf.float32, [None, None, 1], name='utms')

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    grd_matrix, crd_matrix, distance, pred_orien = ConvNext_conv_three_iaff(grd_x, crd_x, keep_prob, is_training)

    loss = compute_loss(distance)

    g_height, g_width, g_channel =grd_matrix.get_shape().as_list()[1:]
    c_height, c_width, c_channel = crd_matrix.get_shape().as_list()[1:]
    grd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])
    crd_global_matrix = np.zeros([input_data.get_test_dataset_size(), c_height, c_width, c_channel])



    # set training
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # print('load model...')
        #
        # if start_epoch == 0:
        #     load_model_path = "D:\CrowdsourcingGeoLocalization\CrowdsourcingGeoLocalization\Model\Initialize\initial_model.ckpt"
        #     saver.restore(sess, load_model_path)
        # else:
        #
        #     load_model_path = '../Model/polar_' + str(polar) + '/' + data_type + '/' + network_type+ '/' + str(start_epoch - 1) + '/model.ckpt'
        #     saver.restore(sess, load_model_path)
        #
        # print("   Model loaded from: %s" % load_model_path)
        # print('load model...FINISHED')

        # Train
        for epoch in range(start_epoch, number_of_epoch):
            iter = 0
            while True:
                # train
                batch_grd, batch_sat, batch_crd, batch_utm= input_data.next_pair_batch(batch_size)

                if batch_sat is None:
                    break

                global_step_val = tf.train.global_step(sess, global_step)

                feed_dict = {crd_x: batch_crd, grd_x: batch_grd,
                             learning_rate: learning_rate_val, keep_prob: keep_prob_val, utms_x:batch_utm}
                if iter % 20 == 0:
                    _, loss_val = sess.run([train_step, loss], feed_dict=feed_dict)
                    print('global %d, epoch %d, iter %d: triplet_loss : %.4f ' %
                          (global_step_val, epoch, iter, loss_val))

                else:
                    sess.run(train_step, feed_dict=feed_dict)

                iter += 1
                del batch_grd, batch_sat, batch_crd, batch_utm

            model_dir = '../Model/polar_' + str(polar) + '/' + data_type + '/' + network_type + '_'+mat_type + '/' + str(epoch) + '/'

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            save_path = saver.save(sess, model_dir + 'model.ckpt')
            print("Model saved in file: %s" % save_path)

            # ---------------------- validation ----------------------
            print('validate...')
            print('   compute global descriptors')
            input_data.reset_scan()

            val_i = 0
            while True:
                # print('      progress %d' % val_i)
                batch_grd, batch_sat, batch_crd, _,= input_data.next_batch_scan(batch_size)
                if batch_sat is None:
                    break
                feed_dict = {crd_x: batch_crd, grd_x: batch_grd, keep_prob: 1.0}
                sat_matrix_val, crd_matrix_val = sess.run([grd_matrix, crd_matrix], feed_dict=feed_dict)

                grd_global_matrix[val_i: val_i + sat_matrix_val.shape[0], :] = sat_matrix_val
                crd_global_matrix[val_i: val_i + crd_matrix_val.shape[0], :] = crd_matrix_val

                val_i += sat_matrix_val.shape[0]

            print('   compute accuracy')
            sat_descriptor = np.reshape(grd_global_matrix[:, :, :c_width, :], [-1, c_height * c_width * c_channel])
            sat_descriptor = sat_descriptor / np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)

            crd_descriptor = np.reshape(crd_global_matrix, [-1, c_height * c_width * c_channel])

            dist_array = 2 - 2 * np.matmul(crd_descriptor, sat_descriptor.transpose())
            val_accuracy = validate(dist_array, 1)

            print('   %d: accuracy = %.1f%%' % (epoch, val_accuracy * 100.0))
            with open('D:/CrowdsourcingGeoLocalization/CrowdsourcingGeoLocalization/Result/' + data_type + '/FOV_polar_' + str(polar) + '_' + str(network_type) + str(mat_type)+ '.txt', 'a+') as file:
                file.write(str(epoch) + ' ' + str(iter) + ' : ' + str(val_accuracy) + '\n')


if __name__ == '__main__':
    train(start_epoch)
