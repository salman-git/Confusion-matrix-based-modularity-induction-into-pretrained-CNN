import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import pickle

class CustomeConv2D(tf.keras.layers.Layer):
    def __init__(self, index, units=32, kernel_size=3, strides=1, padding='same', in_channel=1, activation=tf.nn.relu, batch_normalization=True):
        super(CustomeConv2D, self).__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.in_channel = in_channel
        self.activation = activation
        self.index = index
        # self.filters = tf.Variable(filters, trainable=True)
        # self.biases = tf.Variable(biases, trainable=True)
        self.counter = 0

    def build(self, input_shape):
        
        self.kernel = self.add_weight(shape=(self.kernel_size, self.kernel_size, int(input_shape[-1]), self.units),
                            dtype='float32',
                            name='filter',
                            initializer='random_normal',
                            trainable=True)
        self.bias = self.add_weight(shape=(self.units,),
                                    dtype='float32',
                                    name='bias',
                                    initializer='random_normal',
                                    trainable=True)

    def call(self, inputs, axis=3, mask=None, pre_activations=False):
        x = tf.nn.conv2d(input=inputs, filters=self.kernel, padding=self.padding)
        x = tf.nn.batch_normalization(x, tf.math.reduce_mean(x), tf.math.reduce_variance(x), scale=None, offset=None, variance_epsilon=0.1)
        output = self.activation(x + self.bias)
        
        if pre_activations: 
            return (output, x)
        else: return output