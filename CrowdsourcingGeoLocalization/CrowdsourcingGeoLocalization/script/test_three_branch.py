import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from cir_net_FOV import *
from distance import *
from OriNet_MSHKED.input_data_hk_polar_three import InputData
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import argparse
from tensorflow.python.ops.gen_math_ops import *
import scipy.io as scio
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='TensorFlow implementation.')
import scipy.misc

parser.add_argument('--network_type', type=str, help='network type', default='ConvNext_conv_three_iaff')

parser.add_argument('--start_epoch', type=int, help='from epoch', default=0)
parser.add_argument('--number_of_epoch', type=int, help='number_of_epoch', default=300)
parser.add_argument('--polar', type=int, help='0 or 1', default=1)


args = parser.parse_args()


# --------------  configuration parameters  -------------- #
network_type = args.network_type

start_epoch = args.start_epoch
polar = args.polar
mat_type = 'matdata_HK_full.mat'

number_of_epoch = args.number_of_epoch

data_type = 'MSHKED'

loss_type = 'l1'

batch_size = 64
is_training = False
loss_weight = 5.0
# number_of_epoch = 100

learning_rate_val = 5e-5
keep_prob_val = 0.8

dimension = 4


def block_matrix_multiplication(A, B, block_size=1024):
    """
    Perform matrix multiplication on large matrices by dividing them into blocks.

    Parameters:
    A (numpy.ndarray): The first input matrix.
    B (numpy.ndarray): The second input matrix.
    block_size (int): The size of the blocks to use for the multiplication.

    Returns:
    numpy.ndarray: The result of the matrix multiplication.
    """
    m, k = A.shape
    k, n = B.shape

    C = np.zeros((m, n), dtype=A.dtype)

    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            block_A = A[i:i + block_size, :]
            block_B = B[:, j:j + block_size]
            C[i:i + block_size, j:j + block_size] = np.matmul(block_A, block_B)

    return C

# -------------------------------------------------------- #

if __name__ == '__main__':
    tf.reset_default_graph()

    # import data
    input_data = InputData(polar)

    # define placeholders

    crd_x = tf.placeholder(tf.float32, [None, 128, 170, 3], name='crd_x')
    grd_x = tf.placeholder(tf.float32, [None, 128, 512, 3], name='grd_x')
    sat_x = tf.placeholder(tf.float32, [None, 335, 335, 3], name='sat_x')
    polar_sat_x = tf.placeholder(tf.float32, [None, 128, 512, 3], name='polar_sat_x')



    utms_x = tf.placeholder(tf.float32, [None, None, 1], name='utms')

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    # build model
    mul_matrix, crd_matrix, distance, pred_orien = ConvNext_conv_three_iaff(crd_x, polar_sat_x, grd_x, keep_prob, is_training)

    s_height, s_width, s_channel = mul_matrix.get_shape().as_list()[1:]
    g_height, g_width, g_channel = crd_matrix.get_shape().as_list()[1:]
    mul_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
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
        load_model_path = '../Model/polar_' + str(polar) + '/' + data_type + '/' + network_type + mat_type +'/' + '213' + '/' + 'model.ckpt'

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
            batch_crd, batch_sat_polar, batch_sat, batch_grd, _= input_data.next_batch_scan(batch_size)
            if batch_sat is None:
                break
            feed_dict = {crd_x: batch_crd, grd_x: batch_grd, polar_sat_x: batch_sat_polar, keep_prob: 1.0}
            mul_matrix_val, crd_matrix_val = sess.run([mul_matrix, crd_matrix], feed_dict=feed_dict)

            mul_global_matrix[val_i: val_i + mul_matrix_val.shape[0], :] = mul_matrix_val
            crd_global_matrix[val_i: val_i + crd_matrix_val.shape[0], :] = crd_matrix_val
            #orientation_gth[val_i: val_i + crd_matrix_val.shape[0]] = batch_orien
            val_i += mul_matrix_val.shape[0]

        print('   compute accuracy')
        crd_descriptor = crd_global_matrix
        mul_descriptor = mul_global_matrix

        descriptor_dir = '../Result/MSHKED/Descriptor/'
        if not os.path.exists(descriptor_dir):
            os.makedirs(descriptor_dir)

        # file = descriptor_dir \
        #        + 'train_crd_noise_' + str(train_crd_noise) + '_train_crd_FOV_' + str(train_crd_FOV) \
        #        + 'test_crd_noise_' + str(test_crd_noise) + '_test_crd_FOV_' + str(test_crd_FOV) \
        #        + '_' + network_type + '.mat'
        # scio.savemat(file, {'grd_descriptor': grd_descriptor, 'sat_descriptor': sat_descriptor,
        #                     'orientation_gth': orientation_gth})

        data_amount = crd_descriptor.shape[0]
        top1_percent = int(data_amount * 0.01)

        #if test_crd_noise==0:
        mul_descriptor = np.reshape(mul_global_matrix[:, :, :g_width, :], [-1, g_height * g_width * g_channel])
        mul_descriptor = mul_descriptor / np.linalg.norm(mul_descriptor, axis=-1, keepdims=True)

        crd_descriptor = np.reshape(crd_global_matrix, [-1, g_height * g_width * g_channel])
        nt_mul=np.transpose(mul_descriptor)
        nt_mul=nt_mul.astype(np.float32)
        crd_descriptor=crd_descriptor.astype(np.float32)


        dist_array = 2 - 2 * block_matrix_multiplication(crd_descriptor, nt_mul)


        gt_dist = dist_array.diagonal()
        prediction = np.sum(dist_array < gt_dist.reshape(-1, 1), axis=-1)
        loc_acc = np.sum(prediction.reshape(-1, 1) <= np.arange(10), axis=0) / data_amount

        print(loc_acc)

        dist_array=dist_array.astype(np.int32)
        top50_indices = np.argsort(dist_array, axis=1)[:, :10]

        indices_with_top50 = np.column_stack((np.arange(len(prediction)), top50_indices))

        np.savetxt('indices_with_top50_iaff_all12w.txt', indices_with_top50, fmt='%d')