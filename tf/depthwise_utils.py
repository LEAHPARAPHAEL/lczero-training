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


'''
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

def getFilter(mask_descriptor, channels):
    rook_channels = mask_descriptor[0]
    bishop_channels = mask_descriptor[1]
    knight_channels = mask_descriptor[2]

    if rook_channels > 0:
        r = rookFilter()
        r = np.expand_dims(r, axis=2)
        r = np.repeat(r, rook_channels, axis = 2)
    
    if bishop_channels > 0:
        b = bishopFilter()
        b = np.expand_dims(b, axis=2)
        b = np.repeat(b, bishop_channels, axis = 2)
    
    if knight_channels > 0:
        k  = knightFilter()
        k = np.expand_dims(k , axis=2)
        k  = np.repeat(k , knight_channels, axis = 2)

    if rook_channels > 0 and bishop_channels > 0 and knight_channels > 0:
        m = np.concatenate([r,b,k], axis=2)
    elif rook_channels > 0 and bishop_channels > 0:
        m = np.concatenate([r,b], axis=2)
    elif rook_channels > 0 and knight_channels > 0:
        m = np.concatenate([r,k], axis=2)
    elif bishop_channels > 0 and knight_channels > 0:
        m = np.concatenate([b,k], axis=2)
    elif rook_channels > 0:
        m = r
    elif bishop_channels > 0:
        m = b   
    elif knight_channels > 0:
        m = k      
    
    m = np.expand_dims(m, axis=3)

    return m

class ChessDepthwiseConv2D(Conv2D):
    def __init__(self,
                 mask,
                 kernel_size,
                 strides=(1, 1),
                 precision=tf.float32,
                 padding='same',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 #scale=False,
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
        self.mask_descriptor = mask
        self.precision = precision
        self.padding = self.padding.upper()
        #self.scale = scale

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

        if self.mask_descriptor:
            m = getFilter(self.mask_descriptor, input_dim)
            self.mask = tf.constant(m, dtype=self.precision)
        '''
        if self.scale:
            self.gamma = self.add_weight(
                shape=(1, 1, int(input_shape[channel_axis]), 1),
                initializer='ones',
                name='gamma',
                trainable=True
            )
        '''
        self.built = True

    def call(self, inputs):
        # 1. Apply Mask to Kernel
        kernel = self.depthwise_kernel
        if hasattr(self, "mask"):
            kernel = kernel * self.mask
        '''
        if self.scale:
            spatial_norm = tf.sqrt(tf.reduce_sum(tf.square(kernel), axis=[0, 1], keepdims=True) + 1e-5)
            kernel = kernel / spatial_norm * self.gamma
        '''
        df = "NCHW" if self.data_format == 'channels_first' else "NHWC"
        
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
    
    def get_mask_descriptor(self):
        return getattr(self, "mask_descriptor", None)




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



def get_fused_chess_mask(precision=tf.float32):
    r = np.array(rookFilter(), dtype=np.float32)
    b = np.array(bishopFilter(), dtype=np.float32)
    k = np.array(knightFilter(), dtype=np.float32)

    m = np.stack([r, b, k], axis=-1)
    
    m = np.expand_dims(m, axis=2)
    
    return tf.constant(m, dtype=precision)

class FusedChessDepthwiseConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 strides=(1, 1),
                 padding='same',
                 activation='mish', 
                 use_bias=True,
                 data_format='channels_last',  # Added format support
                 precision=tf.float32,
                 depthwise_initializer='glorot_uniform',
                 recombination_initializer='ones', 
                 **kwargs):
        super(FusedChessDepthwiseConv2D, self).__init__(**kwargs)
        
        self.strides = strides
        self.padding = padding.upper()
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.data_format = data_format
        self.precision = precision
        self.depthwise_initializer = tf.keras.initializers.get(depthwise_initializer)
        self.recombination_initializer = tf.keras.initializers.get(recombination_initializer)

    def build(self, input_shape):
        # 1. Dynamically identify the channel axis
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
            
        self.channels = int(input_shape[channel_axis])

        # Depthwise kernels are ALWAYS [H, W, In_Channels, Multiplier] in TF
        self.depthwise_kernel = self.add_weight(
            shape=(5, 5, self.channels, 1), 
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            trainable=True
        )
        
        # 2. Store weights as (C, 3) to remain format-agnostic in memory
        self.recombination_weights = self.add_weight(
            shape=(self.channels, 3),
            initializer=self.recombination_initializer,
            name='recombination_weights',
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer='zeros',
                name='bias',
                trainable=True
            )
        else:
            self.bias = None
            
        self.mask = get_fused_chess_mask(self.precision)
        self.built = True

    def call(self, inputs):
        masked_kernel = self.depthwise_kernel * self.mask
        
        # 3. Setup format strings and strides
        if self.data_format == 'channels_first':
            df = 'NCHW'
            strides_4d = [1, 1, self.strides[0], self.strides[1]]
        else:
            df = 'NHWC'
            strides_4d = [1, self.strides[0], self.strides[1], 1]
 
        # 4. Execute Depthwise Conv (Expanding internally to 3x)
        x = tf.nn.depthwise_conv2d(
            inputs,
            masked_kernel,
            strides=strides_4d,
            padding=self.padding,
            data_format=df
        )
        
        # 5. Inner Activation (Applied to distinct Rook/Bishop/Knight sums)
        if self.activation is not None:
            x = self.activation(x)
            
        # 6. Dynamic Reshape and Recombine based on data format
        shape = tf.shape(x)
        
        if self.data_format == 'channels_first':
            # x shape: (B, C * 3, H, W) -> Reshape to isolate multiplier: (B, C, 3, H, W)
            x_reshaped = tf.reshape(x, [shape[0], self.channels, 3, shape[2], shape[3]])
            # Broadcast weights to match: (1, C, 3, 1, 1)
            w_reshaped = tf.reshape(self.recombination_weights, [1, self.channels, 3, 1, 1])
            # Multiply and collapse the multiplier axis (axis=2)
            x_recombined = tf.reduce_sum(x_reshaped * w_reshaped, axis=2)
            
        else:
            # x shape: (B, H, W, C * 3) -> Reshape to isolate multiplier: (B, H, W, C, 3)
            x_reshaped = tf.reshape(x, [shape[0], shape[1], shape[2], self.channels, 3])
            # Broadcast weights to match: (1, 1, 1, C, 3)
            w_reshaped = tf.reshape(self.recombination_weights, [1, 1, 1, self.channels, 3])
            # Multiply and collapse the multiplier axis (axis=-1)
            x_recombined = tf.reduce_sum(x_reshaped * w_reshaped, axis=-1)
        
        # 7. Add Bias
        if self.use_bias:
            x_recombined = tf.nn.bias_add(x_recombined, self.bias, data_format=df)
            
        return x_recombined


class ExpandedChessDepthwiseConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 strides=(1, 1),
                 padding='same',
                 activation='mish', 
                 use_bias=True,
                 data_format='channels_last', # Added data_format support
                 precision=tf.float32,
                 depthwise_initializer='glorot_uniform',
                 **kwargs):
        super(ExpandedChessDepthwiseConv2D, self).__init__(**kwargs)
        
        self.strides = strides
        self.padding = padding.upper()
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.data_format = data_format
        self.precision = precision
        self.depthwise_initializer = tf.keras.initializers.get(depthwise_initializer)

    def build(self, input_shape):
        # 1. Dynamically identify the channel axis
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
            
        self.channels = int(input_shape[channel_axis])

        self.depthwise_kernel = self.add_weight(
            shape=(5, 5, self.channels, 3),
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels * 3,),
                initializer='zeros',
                name='bias',
                trainable=True
            )
        else:
            self.bias = None
            
        self.mask = get_fused_chess_mask(self.precision)
        self.built = True

    def call(self, inputs):
        masked_kernel = self.depthwise_kernel * self.mask
        
        # 2. Map Keras string format to TensorFlow low-level string format
        if self.data_format == 'channels_first':
            df = 'NCHW'
            # Batch, Channels, Height, Width
            strides_4d = [1, 1, self.strides[0], self.strides[1]]
        else:
            df = 'NHWC'
            # Batch, Height, Width, Channels
            strides_4d = [1, self.strides[0], self.strides[1], 1]
 
        # 3. Execute with dynamic formatting
        x = tf.nn.depthwise_conv2d(
            inputs,
            masked_kernel,
            strides=strides_4d,
            padding=self.padding,
            data_format=df
        )
        
        # tf.nn.bias_add natively understands 'NCHW' and 'NHWC' broadcasting
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format=df)
            
        if self.activation is not None:
            x = self.activation(x)
            
        return x