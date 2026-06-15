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



'''
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
'''

class ApplySqueezeExcitation(tf.keras.layers.Layer):

    def __init__(self, data_format='channels_first', **kwargs):
        super(ApplySqueezeExcitation, self).__init__(**kwargs)
        # Accept standard Keras data_format string configurations
        self.data_format = data_format

    def build(self, input_dimens):
        # input_dimens[1] is the shape of 'excited': [None, 2 * channels]
        self.reshape_size = input_dimens[1][1]
        self.channels = self.reshape_size // 2

    def call(self, inputs):
        x = inputs[0]
        excited = inputs[1]
        
        # 1. Split first while the tensor is a layout-agnostic 2D vector [None, 2 * channels]
        gammas, betas = tf.split(excited, 2, axis=-1)
        
        # 2. Reshape conditionally based on the target broadcasting layout
        if self.data_format == 'channels_first':
            # Target shape for NCHW stream broadcasting: [None, C, 1, 1]
            gammas = tf.reshape(gammas, [-1, self.channels, 1, 1])
            betas = tf.reshape(betas, [-1, self.channels, 1, 1])
        else:
            # Target shape for NHWC stream broadcasting: [None, 1, 1, C]
            gammas = tf.reshape(gammas, [-1, 1, 1, self.channels])
            betas = tf.reshape(betas, [-1, 1, 1, self.channels])
            
        return tf.nn.sigmoid(gammas) * x + betas




from tensorflow.python.framework import ops

# 1. Load the native compiled binary module
_depthwise_module = tf.load_op_library('./custom_ops/depthwise.so')
fused_chess_depthwise = _depthwise_module.fused_chess_depthwise

# 2. Register the C++ Backward Pass to TensorFlow's Auto-Diff Engine
@ops.RegisterGradient("FusedChessDepthwise")
def _fused_chess_depthwise_grad(op, d_out):
    """
    Connects the native C++ gradient kernel directly to the computational graph.
    """
    rook_threshold = op.get_attr("rook_threshold")
    bishop_threshold = op.get_attr("bishop_threshold")
    knight_threshold = op.get_attr("knight_threshold")
    activation_mode = op.get_attr("activation_mode")  # Extract activation mode attribute
    
    x = op.inputs[0]
    weights = op.inputs[1]
    biases = op.inputs[2]
    
    # Invoke the compiled C++ backward kernel asynchronously
    d_x, d_w, d_b = _depthwise_module.fused_chess_depthwise_grad(
        d_out, x, weights, biases,
        rook_threshold=rook_threshold, bishop_threshold=bishop_threshold, knight_threshold=knight_threshold,
        activation_mode=activation_mode  # Pass to backward pass
    )
    return d_x, d_w, d_b


# 3. Keras High-Level Layer Interface
class FusedChessDepthwiseLayer(tf.keras.layers.Layer):
    def __init__(self, channels, rook_c, bishop_c, knight_c, activation=None, initializer = 'glorot_normal', **kwargs):
        """
        Signature matches your block wrapper initialization footprint:
        du.FusedChessDepthwiseLayer(dff, mask[0], mask[1], mask[2], activation=None, name=...)
        """
        super().__init__(**kwargs)
        self.channels = channels
        self.rook_channels = int(rook_c)
        self.bishop_channels = int(bishop_c)
        self.knight_channels = int(knight_c)
        self.rook_threshold = self.rook_channels
        self.bishop_threshold = self.rook_threshold + self.bishop_channels
        self.knight_threshold = self.bishop_threshold + self.knight_channels
        
        self.initializer = initializer
        # Parse activation mode into integer flags for the C++ backend
        self.activation_str = str(activation).lower() if activation is not None else "none"
        if self.activation_str in ["none", "linear"]:
            self.activation_mode = 0
        elif self.activation_str == "mish":
            self.activation_mode = 1
        else:
            raise ValueError(f"Activation mode '{activation}' is not supported by FusedChessDepthwiseLayer.")
        
    def build(self, input_shape):
        # Resolve active channel dimension dynamically from the incoming stream
        self.channels = input_shape[-1]
        
        # Convolution spatial filters initialized with glorot_normal variance bounds
        self.depthwise_kernel = self.add_weight(
            shape=(9, self.channels),
            initializer=self.initializer,
            trainable=True,
            name='depthwise_kernel'
        )
        
        # Additive channel biases initialized cleanly to zeros
        self.bias = self.add_weight(
            shape=(self.channels,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        
        super().build(input_shape)

    def call(self, inputs):
        """
        Executes on the 3D row-major layout [N, 64, C] with internal FP32 accumulation.
        """
        # Down-cast weights and inputs to float16 tracking variables
        x_fp16 = tf.cast(inputs, tf.float16)
        w_fp16 = tf.cast(self.depthwise_kernel, tf.float16)
        b_fp16 = tf.cast(self.bias, tf.float16)
        
        # Invoke the native operator
        output = fused_chess_depthwise(
            x_fp16, w_fp16, b_fp16, 
            rook_threshold=self.rook_threshold, 
            bishop_threshold=self.bishop_threshold, 
            knight_threshold=self.knight_threshold,
            activation_mode=self.activation_mode  # Pass to forward pass
        )
        
        # Match output precision to the rest of the network's training policy
        return tf.cast(output, inputs.dtype)

    def get_config(self):
        """
        Ensures serialization stability for Keras model checkpoint saving (.keras / .h5)
        """
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "rook_c": self.rook_channels,
            "bishop_c": self.bishop_channels,
            "knight_c": self.knight_channels,
            "activation": self.activation_str,
        })
        return config

    def get_mask_descriptor(self):
        return [self.rook_channels, self.bishop_channels, self.knight_channels]




_depthwise_x_module = tf.load_op_library(os.path.join(loc, 'depthwise_x.so'))

depthwise_x = _depthwise_x_module.depthwise_x
depthwise_x_grad = _depthwise_x_module.depthwise_x_grad


@ops.RegisterGradient("DepthwiseX")
def _depthwise_x_grad(op, d_out):
    activation_mode = op.get_attr("activation_mode")
    
    input_tensor = op.inputs[0]
    weights_tensor = op.inputs[1]
    recomb_tensor = op.inputs[2]
    biases_tensor = op.inputs[3]
    
    d_in, d_w, d_r, d_b = depthwise_x_grad(
        d_out, input_tensor, weights_tensor, recomb_tensor, biases_tensor,
        activation_mode=activation_mode
    )
    return d_in, d_w, d_r, d_b


_depthwise_x_module = tf.load_op_library(os.path.join(loc, 'depthwise_x.so'))

depthwise_x = _depthwise_x_module.depthwise_x
depthwise_x_grad = _depthwise_x_module.depthwise_x_grad

@ops.RegisterGradient("DepthwiseX")
def _depthwise_x_grad(op, d_out):
    activation_mode = op.get_attr("activation_mode")
    
    input_tensor = op.inputs[0]
    weights_tensor = op.inputs[1]
    recomb_tensor = op.inputs[2]
    biases_tensor = op.inputs[3]
    
    d_in, d_w, d_r, d_b = depthwise_x_grad(
        d_out, input_tensor, weights_tensor, recomb_tensor, biases_tensor,
        activation_mode=activation_mode
    )
    return d_in, d_w, d_r, d_b

class DepthwiseXLayer(tf.keras.layers.Layer):
    def __init__(self, channels, activation="mish", name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.channels = channels
        self.activation_str = activation
        self.activation_mode = 1 if activation.lower() == "mish" else 0

    def build(self, input_shape):
        # 27 spatial weights per channel (9 Rook + 9 Bishop + 9 Knight)
        self.depthwise_kernel = self.add_weight(
            shape=(27, self.channels),
            initializer='glorot_normal',
            trainable=True,
            name='depthwise_kernel'
        )
        
        # Absolute Grid Coordinate Modulation Array: 6 items (r, b, k, rb, rk, bk) per board cell
        self.recomb_grid = self.add_weight(
            shape=(8, 8, 6, self.channels),
            initializer='ones', # Initialized to 1.0 to retain symmetric flow features initially
            trainable=True,
            name='recomb_grid'
        )
        
        self.bias = self.add_weight(
            shape=(self.channels,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        
        super().build(input_shape)

    def call(self, inputs):
        x_fp16 = tf.cast(inputs, tf.float16)
        w_fp16 = tf.cast(self.depthwise_kernel, tf.float16)
        r_fp16 = tf.cast(self.recomb_grid, tf.float16)
        b_fp16 = tf.cast(self.bias, tf.float16)
        
        output = depthwise_x(
            x_fp16, w_fp16, r_fp16, b_fp16,
            activation_mode=self.activation_mode
        )
        return tf.cast(output, inputs.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "activation": self.activation_str
        })
        return config