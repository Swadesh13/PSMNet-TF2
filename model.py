import tensorflow as tf
import tensorflow.keras as keras
from model_utils import *


class CNN(keras.layers.Layer):
    def __init__(self, reg, **kwargs):
        super(CNN, self).__init__(name = 'CNN', **kwargs)
        self.conv0_x = []
        self.conv1_x = []
        self.conv2_x = []
        self.conv3_x = []
        self.conv4_x = []

        self.conv0_x.append(ConvBlock('conv2d', 32, 3, 2, name='conv0_1', reg=reg))
        for i in range(1, 3):
            self.conv0_x.append(ConvBlock('conv2d', 32, 3, name='conv0_%d' % (i+1), reg=reg))
        for i in range(3):
            self.conv1_x.append(ResBlock('conv2d', 32, 3, name='conv1_%d' % (i+1), reg=reg))
        self.conv2_x.append(ResBlock('conv2d', 64, 3, strides=2, name='conv2_1', reg=reg, projection=True))
        for i in range(1, 8):
            self.conv2_x.append(ResBlock('conv2d', 64, 3, name='conv2_%d' % (i+1), reg=reg))
        self.conv3_x.append(ResBlock('conv2d', 128, 3, dilation_rate=2, name='conv3_1', reg=reg, projection=True))
        for i in range(1, 3):
            self.conv3_x.append(ResBlock('conv2d', 128, 3, dilation_rate=2, name='conv3_%d' % (i+1), reg=reg))
        for i in range(3):
            self.conv4_x.append(ResBlock('conv2d', 128, 3, dilation_rate=4, name='conv4_%d' % (i+1), reg=reg))
    
    def call(self, x: tf.Tensor):
        # Output also includes the 2_8 output for the SPP network.
        for layer in [*self.conv0_x, *self.conv1_x, *self.conv2_x]:
            x = layer(x)
        y = x
        for layer in [*self.conv3_x, *self.conv4_x]:
            x = layer(x)
        return [y, x]

class SPP(keras.layers.Layer):
    def __init__(self, reg, **kwargs):
        super(SPP, self).__init__(name = 'SPP', **kwargs)
        self.branches = []
        for i, p in enumerate([64, 32, 16, 8]):
            self.branches.append(SPPBranch('conv2d', p, 32, 3, name='branch_%d' % (i+1), reg=reg))
        self.conv1 = ConvBlock('conv2d', 128, 3, name='fusion_conv1', reg=reg)
        self.conv2 = ConvBlock('conv2d', 32, 1, name='fusion_conv2', reg=reg)
    
    def call(self, x):
        conv2_8, x = x
        self.outputs = []
        for layer in self.branches:
            self.outputs.append(layer(x))
        concat = tf.concat([conv2_8, x] + self.outputs, axis=-1, name='concat')
        fusion = self.conv1(concat)
        fusion = self.conv2(fusion)
        return fusion

class CNN3D(keras.layers.Layer):
    def __init__(self, reg, **kwargs):
        super(CNN3D, self).__init__('CNN3D', **kwargs)
        self.conv_layers = []
        for i in range(6):
            self.conv_layers.append(ConvBlock('conv3d', 32, 3, name='3Dconv0_%d' % (i+1), reg=reg))
        self.conv_layers.append(ConvBlock('conv3d', 1, 3, name='3Dconv5', reg=reg))
    
    def call(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

class ResNet3D(keras.layers.Layer):
    def __init__(self, reg, **kwargs):
        super(ResNet3D, self).__init__(name = 'ResNet3D', **kwargs)
        self.conv0 = []
        for i in range(2):
            self.conv0.append(ConvBlock('conv3d', 32, 3, name='3Dconv0_%d' % (i+1), reg=reg))
        self.conv1 = ResBlock('conv3d', 32, 3, name='3Dconv1', reg=reg)
        self.conv2 = ResBlock('conv3d', 32, 3, name='3Dconv2', reg=reg)
        self.conv3 = ResBlock('conv3d', 32, 3, name='3Dconv3', reg=reg)
        self.conv4 = ResBlock('conv3d', 32, 3, name='3Dconv4', reg=reg)
        self.out = ConvBlock('conv3d', 1, 3, name='3Dconv5', reg=reg)

    def call(self, x):
        for layer in self.conv0:
            x = layer(x)
        _3Dconv1 = self.conv1(x)
        _3Dconv1 = self.conv2(_3Dconv1)
        _3Dconv1 = self.conv3(_3Dconv1)
        _3Dconv1 = self.conv4(_3Dconv1)
        out = self.out(_3Dconv1)
        return out

class DenseNet3D(keras.layers.Layer):
    def __init__(self, reg, **kwargs):
        super(DenseNet3D, self).__init__('DenseNet3D', **kwargs)
        self.conv1 = []
        for i in range(2):
            self.conv_layers.append(ConvBlock('conv3d', 32, 3, name='3Dconv0_%d' % (i+1), reg=reg))
        self.conv2 = []
        for j in range(5):
            self.conv2.append(ConvBlock('conv3d', 32, 3, name='3Dconv1_%d' % (j+1), reg=self.reg))
        self.out = ConvBlock('conv3d', 1, 3, name='3Dconv5', reg=self.reg)

    def call(self, x):
        for layer in self.conv1:
            x = layer(x)
        out = [x]
        for layer in self.conv2:
            x = layer(x)
            out.append(x)
            x = tf.concat(out, axis=-1)

        return self.out(x)

@tf.function
def cost_vol(left, right, max_disp=192):
    shape = tf.shape(right)
    right_tensor = keras.backend.spatial_2d_padding(right, padding=((0, 0), (max_disp // 4, 0)))
    disparity_costs = []
    for d in reversed(range(max_disp // 4)):
        left_tensor_slice = left
        right_tensor_slice = tf.slice(right_tensor, begin=[0, 0, d, 0], size=shape)
        right_tensor_slice.set_shape(tf.TensorShape([None, None, None, 32]))
        cost = tf.concat([left_tensor_slice, right_tensor_slice], axis=3)
        disparity_costs.append(cost)
    cost_vol = tf.stack(disparity_costs, axis=1)
    return cost_vol

@tf.function
def output(output, image_size, max_disp, height):
    squeeze = tf.squeeze(output, [4])
    transpose = tf.transpose(squeeze, [0, 2, 3, 1])

    upsample = tf.transpose(tf.image.resize(transpose, image_size), [0, 3, 1, 2])
    upsample = tf.image.resize(upsample, tf.constant([max_disp, height], dtype=tf.int32))
    disps = soft_arg_min(upsample)
    return disps


class Model(keras.Model):
    def __init__(self, height=256, width=512, batch_size=4, max_disp=128, lr=0.001, cnn_3d_type='resnet_3d', reg=1e-4, **kwargs):
        super(Model, self).__init__(name='PSMNet', **kwargs)
        self.reg = reg
        self.max_disp = max_disp
        self.image_size_tf = None
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.lr = lr
        self.cnn_3d = cnn_3d_type

        self.conv4 = CNN(reg)
        self.fusion = SPP(reg)
        if self.cnn_3d == 'normal':
            self.conv3D_block = CNN3D(reg)
        elif self.cnn_3d == 'resnet_3d':
            self.conv3D_block = ResNet3D(reg)
        elif self.cnn_3d == 'densenet_3d':
            self.conv3D_block = DenseNet3D(reg)
        else:
            raise NotImplementedError('Does not support {}'.format(self.cnn_3d))

    def call(self, x):
        left, right = x
        self.image_size = tf.shape(left)[1:3]
        conv4_left = self.conv4(left)
        fusion_left = self.fusion(conv4_left)
        conv4_right = self.conv4(right)
        fusion_right = self.fusion(conv4_right)
        cv = cost_vol(fusion_left, fusion_right, self.max_disp)
        outputs = self.conv3D_block(cv)
        disps = output(outputs, self.image_size, self.max_disp, self.height)
        return disps

if __name__ == '__main__':
    model = Model()
    disps = model(tf.ones((2, 4, 256, 512, 3), dtype=tf.float64))
    print(disps)