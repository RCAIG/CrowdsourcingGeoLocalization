import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from cir_net_FOV import *
from distance import *
from input_data import InputData
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
mat_type = 'metadata.mat'

number_of_epoch = args.number_of_epoch

data_type = ''

loss_type = 'l1'

batch_size = 64
is_training = False
loss_weight = 5.0

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


def validate_with_utm_threshold(dist_array, utm_coords, threshold_meters=25.0, topK_list=[1, 5, 10]):
    """
    Calculate Recall@N metrics based on UTM coordinates and distance threshold
    
    Args:
        dist_array: Distance matrix [N, N]
        utm_coords: UTM coordinates [N, 2] (x, y)
        threshold_meters: Distance threshold in meters
        topK_list: List of top-K values to calculate, e.g. [1, 5, 10]
    
    Returns:
        recall_scores: Recall scores for each top-K
    """
    N = dist_array.shape[0]
    recall_scores = {}
    
    for topK in topK_list:
        correct_count = 0
        
        for i in range(N):
            # Get UTM coordinates for current query
            query_utm = utm_coords[i]
            
            # Get top-K most similar candidates
            top_k_indices = np.argsort(dist_array[i, :])[:topK]
            
            # Check if any of the top-K candidates are within threshold
            is_correct = False
            for candidate_idx in top_k_indices:
                candidate_utm = utm_coords[candidate_idx]
                # Calculate Euclidean distance in meters
                distance = np.sqrt((query_utm[0] - candidate_utm[0])**2 + 
                                 (query_utm[1] - candidate_utm[1])**2)
                if distance <= threshold_meters:
                    is_correct = True
                    break
            
            if is_correct:
                correct_count += 1
        
        recall_scores[f'R@{topK}'] = correct_count / N
    
    return recall_scores

# -------------------------------------------------------- #

if __name__ == '__main__':
    tf.reset_default_graph()

    # Import data
    input_data = InputData(polar)

    # Define placeholders
    crd_x = tf.placeholder(tf.float32, [None, 128, 170, 3], name='crd_x')
    grd_x = tf.placeholder(tf.float32, [None, 128, 512, 3], name='grd_x')
    polar_sat_x = tf.placeholder(tf.float32, [None, 128, 512, 3], name='polar_sat_x')


    utms_x = tf.placeholder(tf.float32, [None, None, 1], name='utms')

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    # Build model
    mul_matrix, crd_matrix, distance, pred_orien = ConvNext_conv_three_iaff(crd_x, polar_sat_x, grd_x, keep_prob, is_training)

    s_height, s_width, s_channel = mul_matrix.get_shape().as_list()[1:]
    g_height, g_width, g_channel = crd_matrix.get_shape().as_list()[1:]
    mul_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
    crd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    global_vars = tf.global_variables()

    # Run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print('load model...')

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
            val_i += mul_matrix_val.shape[0]

        print('   compute accuracy')
        crd_descriptor = crd_global_matrix
        mul_descriptor = mul_global_matrix

        descriptor_dir = '../Result/CVACT/Descriptor/'
        if not os.path.exists(descriptor_dir):
            os.makedirs(descriptor_dir)


        data_amount = crd_descriptor.shape[0]
        top1_percent = int(data_amount * 0.01)

        mul_descriptor = np.reshape(mul_global_matrix[:, :, :g_width, :], [-1, g_height * g_width * g_channel])
        mul_descriptor = mul_descriptor / np.linalg.norm(mul_descriptor, axis=-1, keepdims=True)

        crd_descriptor = np.reshape(crd_global_matrix, [-1, g_height * g_width * g_channel])
        nt_mul=np.transpose(mul_descriptor)
        nt_mul=nt_mul.astype(np.float32)
        crd_descriptor=crd_descriptor.astype(np.float32)


        dist_array = 2 - 2 * block_matrix_multiplication(crd_descriptor, nt_mul)

        # Get UTM coordinates for distance threshold-based evaluation
        utm_coords = input_data.get_utm_coords()
        
        # Use new evaluation metrics
        recall_scores = validate_with_utm_threshold(dist_array, utm_coords, threshold_meters=25.0, topK_list=[1, 5, 10])
        
        print('R@1 = %.1f%%, R@5 = %.1f%%, R@10 = %.1f%%' % 
              (recall_scores['R@1'] * 100.0, recall_scores['R@5'] * 100.0, recall_scores['R@10'] * 100.0))
        # Get top 50 most matching image indices for each image
        dist_array=dist_array.astype(np.int32)
        top50_indices = np.argsort(dist_array, axis=1)[:, :50]

        # Save to text file
        indices_with_top50 = np.column_stack((np.arange(data_amount), top50_indices))

        # Save to text file
        np.savetxt('indices_with_top50_iaff_all12w.txt', indices_with_top50, fmt='%d')



