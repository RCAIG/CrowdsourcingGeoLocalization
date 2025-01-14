import tensorflow as tf


def layernorm(x, epsilon=1e-6):
    return tf.keras.layers.LayerNormalization(epsilon=epsilon)(x)


def depthwise_conv2d(x, kernel_size=7):
    return tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same')(x)


def convnext_block(x, dim, drop_path=0.):
    shortcut = x

    # LayerNorm
    x = layernorm(x)

    # Depthwise Conv
    x = depthwise_conv2d(x)

    # LayerNorm
    x = layernorm(x)

    # Linear projection 1
    x = tf.keras.layers.Dense(4 * dim)(x)
    x = tf.keras.layers.Activation('gelu')(x)

    # Linear projection 2
    x = tf.keras.layers.Dense(dim)(x)

    # Drop Path
    if drop_path > 0:
        x = tf.keras.layers.Dropout(drop_path)(x)

    # Residual connection
    x = tf.keras.layers.Add()([shortcut, x])

    return x


def downsample_layer(x, dim):
    x = layernorm(x)
    x = tf.keras.layers.Conv2D(dim, kernel_size=2, strides=2)(x)
    return x


def feature_adjustment_layer(x, output_channels):
    # Modify the feature map: decrease height while keeping width
    x = tf.keras.layers.Conv2D(output_channels, kernel_size=(3, 1), strides=(1, 1), padding='same')(x)
    return x


def convnext_b(x, keep_prob, num_channels, trainable, name):
    with tf.compat.v1.variable_scope(name):
        # Stem
        x = tf.keras.layers.Conv2D(128, kernel_size=4, strides=4)(x)
        x = layernorm(x)

        # Stage 1
        for _ in range(3):
            x = convnext_block(x, dim=128)

        # Downsampling
        x = downsample_layer(x, 256)

        # Stage 2
        for _ in range(3):
            x = convnext_block(x, dim=256)

        # Downsampling
        x = downsample_layer(x, 512)

        # Stage 3
        for _ in range(27):
            x = convnext_block(x, dim=512)

        # Downsampling
        x = downsample_layer(x, 1024)

        # Stage 4: reduce height feature maps while keeping width
        x = feature_adjustment_layer(x, output_channels=512)  # Example adjustment
        x = feature_adjustment_layer(x, output_channels=16)  # Final output adjustment

        # Final normalization
        x = layernorm(x)

        # Additional convolutional layer to match num_channels
        x = tf.keras.layers.Conv2D(num_channels, 1)(x)

        return x