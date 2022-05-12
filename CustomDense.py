import tensorflow as tf

class CustomDense(tf.keras.layers.Layer):
  
  def __init__(self, num_outputs, activation=None):
    super(CustomDense, self).__init__()
    self.num_outputs = num_outputs
    self.activation = activation

  def build(self, input_shape):
    self.kernel = self.add_weight(name="kernel",
                                    shape=[int(input_shape[-1]),
                                           self.num_outputs],
                                    dtype='float32',
                                    initializer='random_normal',
                                    trainable=True)
    self.bias = self.add_weight(name='bias', shape=(self.num_outputs,), 
                                    dtype='float32',
                                    initializer='random_normal',
                                    trainable=True)

  def call(self, input):
    kernel_slice = self.kernel #if self.kernel.shape == input.shape else self.kernel[0:input.shape[-1]]
    return self.activation(tf.matmul(input, kernel_slice) + self.bias) if self.activation else tf.matmul(input, kernel_slice) + self.bias
    # return self.activation(tf.matmul(input, self.kernel)) if self.activation else tf.matmul(input, self.kernel)
