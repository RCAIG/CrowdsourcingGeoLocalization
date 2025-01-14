import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from cir_net_FOV import *
from distance import *
from OriNet_MSHKED.input_data_hk_polar_crd_sat import InputData
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import argparse
from tensorflow.python.ops.gen_math_ops import *
import scipy.io as scio

parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type', type=str, help='network type', default='ConvNext_conv_three_iaff')

parser.add_argument('--start_epoch', type=int, help='from epoch', default=0)
parser.add_argument('--number_of_epoch', type=int, help='number_of_epoch', default=300)
parser.add_argument('--polar', type=int, help='0 or 1', default=1)

args = parser.parse_args()


# --------------  configuration parameters  -------------- #
network_type = args.network_type

start_epoch = args.start_epoch
polar = args.polar

number_of_epoch = args.number_of_epoch

data_type = 'MSHKED'
mat_type = 'matdata_HK_full_0.2.mat'
loss_type = 'l1'

batch_size = 64
is_training = False
loss_weight = 5.0
# number_of_epoch = 100

learning_rate_val = 5e-5
keep_prob_val = 0.8

dimension = 4


# -------------------------------------------------------- #

if __name__ == '__main__':
    tf.reset_default_graph()

    # import data
    input_data = InputData(polar)

    # define placeholders

    crd_x = tf.placeholder(tf.float32, [None, 128, 170, 3], name='crd_x')
    sat_x = tf.placeholder(tf.float32, [None, 335, 335, 3], name='sat_x')
    polar_sat_x = tf.placeholder(tf.float32, [None, 128, 512, 3], name='polar_sat_x')



    utms_x = tf.placeholder(tf.float32, [None, None, 1], name='utms')

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    # build model
    sat_matrix, crd_matrix, distance, pred_orien = ConvNext_conv_three_iaff(polar_sat_x, crd_x, keep_prob, is_training)

    s_height, s_width, s_channel = sat_matrix.get_shape().as_list()[1:]
    g_height, g_width, g_channel = crd_matrix.get_shape().as_list()[1:]
    sat_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
    crd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])


    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    global_vars = tf.global_variables()

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print('load model...')

        # load_model_path = '../Model/polar_' + str(polar) + '/' + data_type + '/' + network_type \
        #                   + '/train_crd_noise_' + str(train_crd_noise) + '/train_crd_FOV_' + str(train_crd_FOV) \
        #                   + '/model.ckpt'
        load_model_path = '../Model/polar_' + str(polar) + '/' + data_type + '/' + network_type+ '_'+ mat_type+ '/'+'72' + '/' + 'model.ckpt'

        saver.restore(sess, load_model_path)

        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')

        # ---------------------- validation ----------------------
        print('validate...')
        print('   compute global descriptors')
        input_data.reset_scan()
        np.random.seed(2019)

        val_i = 0
        while True:
            print('      progress %d' % val_i)
            batch_sat_polar, batch_sat, batch_crd, _= input_data.next_batch_scan(batch_size)
            if batch_sat is None:
                break
            feed_dict = {crd_x: batch_crd, polar_sat_x: batch_sat_polar, keep_prob: 1.0}
            sat_matrix_val, crd_matrix_val = sess.run([sat_matrix, crd_matrix], feed_dict=feed_dict)

            sat_global_matrix[val_i: val_i + sat_matrix_val.shape[0], :] = sat_matrix_val
            crd_global_matrix[val_i: val_i + crd_matrix_val.shape[0], :] = crd_matrix_val
            #orientation_gth[val_i: val_i + crd_matrix_val.shape[0]] = batch_orien
            val_i += sat_matrix_val.shape[0]

        print('   compute accuracy')
        crd_descriptor = crd_global_matrix
        sat_descriptor = sat_global_matrix

        descriptor_dir = '../Result/MSHKED/Descriptor/'
        if not os.path.exists(descriptor_dir):
            os.makedirs(descriptor_dir)

        # file = descriptor_dir \
        #        + 'train_crd_noise_' + str(train_crd_noise) + '_train_crd_FOV_' + str(train_crd_FOV) \
        #        + 'test_crd_noise_' + str(test_crd_noise) + '_test_crd_FOV_' + str(test_crd_FOV) \
        #        + '_' + network_type + '.mat'
        # scio.savemat(file, {'crd_descriptor': crd_descriptor, 'sat_descriptor': sat_descriptor,
        #                     'orientation_gth': orientation_gth})

        data_amount = crd_descriptor.shape[0]
        top1_percent = int(data_amount * 0.01) + 1

        #if test_crd_noise==0:
        sat_descriptor = np.reshape(sat_global_matrix[:, :, :g_width, :], [-1, g_height * g_width * g_channel])
        sat_descriptor = sat_descriptor / np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)

        crd_descriptor = np.reshape(crd_global_matrix, [-1, g_height * g_width * g_channel])

        dist_array = 2 - 2 * np.matmul(crd_descriptor, np.transpose(sat_descriptor))
        gt_dist = dist_array.diagonal()
        prediction = np.sum(dist_array < gt_dist.reshape(-1, 1), axis=-1)
        correct_indices = np.where(prediction.reshape(-1, 1) < np.arange(2))[0]
        print(correct_indices)
        loc_acc_all = np.sum(prediction.reshape(-1, 1) <= np.arange(10), axis=0)
        print(loc_acc_all)
        loc_acc = loc_acc_all/ data_amount

        print(loc_acc)
        top50_indices = np.argsort(dist_array, axis=1)[:, :10]


        indices_with_top10 = np.column_stack((np.arange(len(prediction)), top50_indices))


        np.savetxt('indices_with_top50_crdsat.txt', indices_with_top10, fmt='%d')
