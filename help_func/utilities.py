from keras.layers import *
from keras.activations import relu
CLASS_NUM = 2

def ReguBlock(n_output):
    # n_output: number of feature maps in the block
    # keras functional api: return the function of type
    # Tensor -> Tensor
    def f(x):

        # H_l(x):
        # first pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # first convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=regularizers.l2(0.0005))(h)

        # second pre-activation
        h = BatchNormalization()(h)
        h = Activation(relu)(h)
        # second convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=regularizers.l2(0.0005))(h)
        return h

    return f