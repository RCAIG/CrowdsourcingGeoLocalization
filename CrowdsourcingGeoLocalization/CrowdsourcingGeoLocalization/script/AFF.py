import tensorflow as tf
from tensorflow.keras import layers

class AFF(tf.keras.Model):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)


        self.local_att = tf.keras.Sequential([
            layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='valid'),#(?, 4, 64, 1)
            layers.BatchNormalization(),#(?, 4, 64, 1)
            layers.ReLU(),#(?, 4, 64, 1)
            layers.Conv2D(channels, kernel_size=1, strides=1, padding='valid'),#(?, 4, 64, 4)
            # layers.BatchNormalization()
        ])


        self.global_att = tf.keras.Sequential([
            layers.AveragePooling2D(pool_size=(4, 64)),
            layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='valid'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(channels, kernel_size=1, strides=1, padding='valid'),
            layers.BatchNormalization()
        ])

        self.sigmoid = tf.keras.activations.sigmoid

    def call(self, x, residual):
        xa = x + residual #(?, 4, 64, 8)
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo