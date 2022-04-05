import tensorflow as tf
import tensorflow.keras as keras


class ConvBlock(keras.layers.Layer):
    """Basic conv-bn-relu function"""
    def __init__(self, func: str, filters, kernel_size, strides=1, dilation_rate=1, name: str = None, reg=1e-4, apply_bn=True, apply_relu=True, **kwargs):
        super(ConvBlock, self).__init__(name = name, **kwargs)
        if func.lower() == "conv2d":
            self.conv_layer = keras.layers.Conv2D(filters, kernel_size, strides, 'same', dilation_rate=dilation_rate, kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(reg), bias_regularizer=keras.regularizers.l2(reg))
        elif func.lower() == "conv3d":
            self.conv_layer = keras.layers.Conv3D(filters, kernel_size, strides, 'same', dilation_rate=dilation_rate, kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(reg), bias_regularizer=keras.regularizers.l2(reg))
        if apply_bn:
            self.batch_norm = keras.layers.BatchNormalization()
        self.apply_bn = apply_bn
        self.apply_relu = apply_relu

    def call(self, x):
        bottom = self.conv_layer(x)
        if self.apply_bn:
            bottom = self.batch_norm(bottom)
        if self.apply_relu:
            return keras.activations.relu(bottom)
        else:
            return bottom

class ResBlock(keras.layers.Layer):
    def __init__(self, func: str, filters, kernel_size, strides=1, dilation_rate=1, name=None, reg=1e-4, projection=False, **kwargs):
        super(ResBlock, self).__init__(name = name, **kwargs)
        self.conv1 = ConvBlock(func, filters, kernel_size, strides, dilation_rate, 'conv1', reg)
        self.conv2 = ConvBlock(func, filters, kernel_size, 1, dilation_rate, 'conv2', reg, apply_relu=False)
        if projection:
            self.short_cut = keras.layers.Conv2D(filters, 1, strides, 'same', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(reg), bias_regularizer=keras.regularizers.l2(reg), name = 'projection')
        self.projection = projection

    def call(self, x):
        short_cut = x
        bottom = self.conv1(x)
        bottom = self.conv2(bottom)
        if self.projection:
            short_cut = self.short_cut(short_cut)
        bottom = tf.add(bottom, short_cut, 'add')
        return bottom

class SPPBranch(keras.layers.Layer):
    def __init__(self, func, pool_size, filters, kernel_size, strides=1, dilation_rate=1, name=None, reg=1e-4, apply_bn=True, apply_relu=True, **kwargs):
        super(SPPBranch, self).__init__(name = name, **kwargs)
        self.conv_block = ConvBlock(func, filters, kernel_size, strides, dilation_rate, 'conv', reg, apply_bn, apply_relu)
        self.avg_pool = keras.layers.AveragePooling2D(pool_size, padding='same', name='avg_pool')
    
    def call(self, x):
        size = tf.shape(x)[1:3]
        bottom = self.conv_block(x)
        bottom = self.avg_pool(x)
        return tf.image.resize(bottom, size)

@tf.function
def soft_arg_min(filtered_cost_volume):
    """Disparity Regression"""
    probability_volume = tf.nn.softmax(tf.scalar_mul(-1, filtered_cost_volume), axis=1, name='prob_volume')
    volume_shape = tf.shape(probability_volume)
    soft_1d = tf.cast(tf.range(0, volume_shape[1], dtype=tf.int32),tf.float32)
    soft_4d = tf.tile(soft_1d, tf.stack([volume_shape[0] * volume_shape[2] * volume_shape[3]]))
    soft_4d = tf.reshape(soft_4d, [volume_shape[0], volume_shape[2], volume_shape[3], volume_shape[1]])
    soft_4d = tf.transpose(soft_4d, [0, 3, 1, 2])
    estimated_disp_image = tf.reduce_sum(soft_4d * probability_volume, axis=1)
    print(estimated_disp_image.shape)
    return estimated_disp_image
