import numpy as np
import os
import tensorflow as tf
import time
import bisect
import attention_policy_map as apm
from functools import reduce
import operator
import functools
from net import Net

from keras import backend as K
from tensorflow.keras.layers import Conv2D

def randomFilter():


    A = np.zeros((5,5))
    
    for i in range(5):
        for j in range(5):
            
            A[i,j] = np.random.randint(2)
            
    print(A)
    
    return A


def randomize_filter(kernel_size, matrix_filter):
    
    nb_activation = np.sum(matrix_filter)-1
    
    A = np.zeros((kernel_size,kernel_size))
    
    A[kernel_size//2,kernel_size//2] = 1
      
      
    while(nb_activation > 0):
        
        pos1 = random.randint(0, kernel_size -1)
        pos2 = random.randint(0, kernel_size -1)
    
        if(A[pos1,pos2] == 0):
            
            A[pos1,pos2] = 1
            
            nb_activation = nb_activation - 1
                        
    
    return A


def queenFilter():


    A = [[1,0,1,0,1],
         [0,1,1,1,0],
         [1,1,1,1,1],
         [0,1,1,1,0],
         [1,0,1,0,1]]

    return A




def bishopFilter():


    A = [[1,0,0,0,1],
         [0,1,0,1,0],
         [0,0,1,0,0],
         [0,1,0,1,0],
         [1,0,0,0,1]]

    return A

def rookFilter():

    A = [[0,0,1,0,0],
         [0,0,1,0,0],
         [1,1,1,1,1],
         [0,0,1,0,0],
         [0,0,1,0,0]]

    return A

def knightFilter():


    A = [[0,1,0,1,0],
         [1,0,0,0,1],
         [0,0,1,0,0],
         [1,0,0,0,1],
         [0,1,0,1,0]]

    return A



def getFilter(mask_type, channels):
    
    #print(mask_type)
    
    if(mask_type == 'rbk'):
        
        b = bishopFilter()
        b = np.expand_dims(b, axis=2)
        b = np.repeat(b, channels//3, axis = 2)
        
        r = rookFilter()
        r = np.expand_dims(r, axis=2)
        r = np.repeat(r, channels//3, axis = 2)
        
        k  = knightFilter()
        k = np.expand_dims(k , axis=2)
        k  = np.repeat(k , channels//3, axis = 2)

        m = np.concatenate([r,b,k], axis=2)
        
    elif(mask_type == 'rb'):
    
        b = bishopFilter()
        b = np.expand_dims(b, axis=2)
        b = np.repeat(b, channels//2, axis = 2)
        
        r = rookFilter()
        r = np.expand_dims(r, axis=2)
        r = np.repeat(r, channels//2, axis = 2)
        
        m = np.concatenate([r,b], axis=2)        
    
    elif(mask_type == 'random'):
    
        b = bishopFilter()
        b = randomize_filter(5,b)
        
        b = np.expand_dims(b, axis=2)
        b = np.repeat(b, channels//3, axis = 2)
        
        r = rookFilter()
        r = randomize_filter(5,r)
        
        r = np.expand_dims(r, axis=2)
        r = np.repeat(r, channels//3, axis = 2)
        
        k  = knightFilter()
        k = randomize_filter(5,k)

        k = np.expand_dims(k, axis=2)
        k = np.repeat(k, channels//3, axis = 2)
        
        m = np.concatenate([b,r,k], axis=2)
        
        
    m = np.expand_dims(m, axis=3)

    return m
'''
class ChessDepthwiseConv2D(Conv2D):

    def __init__(self,
                mask_type,
                kernel_size,
                strides=(1, 1),
                precision=tf.float32,
                padding='same',
                depth_multiplier=1,
                data_format=None,
                activation=None,
                use_bias=True,
                depthwise_initializer='glorot_uniform',
                bias_initializer='zeros',
                depthwise_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                depthwise_constraint=None,
                bias_constraint=None,
                **kwargs):
        super(ChessDepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        
        self.depthwise_initializer = depthwise_initializer
        self.depthwise_regularizer = depthwise_regularizer
        self.depthwise_constraint = depthwise_constraint
        self.bias_initializer = bias_initializer
        self.mask_type = mask_type
        self.precision = precision

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `ChessDepthwiseConv2D` should have rank 4. '
                            'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                            '`ChessDepthwiseConv2D` '
                            'should be defined. Found `None`.')
        
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                self.kernel_size[1],
                                input_dim,
                                self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True
        
        
        channels = input_shape[1]
        
        print("input chess channel")
        print(channels)
            
        print("self.mask_type")
        print(self.mask_type)    
        
        if self.mask_type:
            m = getFilter(self.mask_type, channels)
            print(m.shape)
            self.mask = tf.constant(m, dtype=self.precision)
        
    def call(self, inputs, training=None):
      
        if hasattr(self, "mask"):
            outputs = K.depthwise_conv2d(
                inputs,
                self.depthwise_kernel*self.mask,
                strides=self.strides,
                padding=self.padding,
                dilation_rate=self.dilation_rate,
                data_format=self.data_format)
        else:
            outputs = K.depthwise_conv2d(
                inputs,
                self.depthwise_kernel,
                strides=self.strides,
                padding=self.padding,
                dilation_rate=self.dilation_rate,
                data_format=self.data_format)      

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs
  
    def get_mask(self):
        if hasattr(self, "mask"):
            return self.mask
        return "no mask"
'''


class ChessDepthwiseConv2D(Conv2D):
    def __init__(self,
                 mask_type,
                 kernel_size,
                 strides=(1, 1),
                 precision=tf.float32,
                 padding='same',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ChessDepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding, # tf.nn wants 'SAME' or 'VALID'
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = depthwise_initializer
        self.depthwise_regularizer = depthwise_regularizer
        self.depthwise_constraint = depthwise_constraint
        self.bias_initializer = bias_initializer
        self.mask_type = mask_type
        self.precision = precision
        self.padding = self.padding.upper()

    def build(self, input_shape):
        # Handle Channel Axis
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                self.kernel_size[1],
                                input_dim,
                                self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        if self.mask_type:
            # Assuming getFilter is defined elsewhere in your depthwise_utils.py
            m = getFilter(self.mask_type, input_dim)
            self.mask = tf.constant(m, dtype=self.precision)

        self.built = True

    def call(self, inputs):
        # 1. Apply Mask to Kernel
        kernel = self.depthwise_kernel
        if hasattr(self, "mask"):
            kernel = kernel * self.mask

        # 2. Execute Depthwise Conv using TensorFlow Backend
        # tf.nn expects string format like "NCHW" or "NHWC"
        df = "NCHW" if self.data_format == 'channels_first' else "NHWC"
        
        # tf.nn.depthwise_conv2d uses 4D strides: [1, stride, stride, 1]
        strides_4d = [1, self.strides[0], self.strides[1], 1]

        outputs = tf.nn.depthwise_conv2d(
            inputs,
            kernel,
            strides=strides_4d,
            padding=self.padding,
            data_format=df
        )

        # 3. Add Bias
        if self.bias is not None:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=df)

        # 4. Activation
        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def get_mask(self):
        return getattr(self, "mask", None)
    
    def get_mask_type(self):
        return getattr(self, "mask_type", "")




class ApplySqueezeExcitation(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ApplySqueezeExcitation, self).__init__(**kwargs)

    def build(self, input_dimens):
        self.reshape_size = input_dimens[1][1]

    def call(self, inputs):
        x = inputs[0]
        excited = inputs[1]
        gammas, betas = tf.split(tf.reshape(excited,
                                            [-1, self.reshape_size, 1, 1]),
                                 2,
                                 axis=1)
        return tf.nn.sigmoid(gammas) * x + betas


