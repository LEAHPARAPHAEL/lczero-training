import numpy as np
import tensorflow as tf

def neighbourMask3x3():
    return np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]], dtype=np.float32)

def diagonalMask3x3():
    return np.array([[1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 1]], dtype=np.float32)

def neighbourMask5x5():
    return np.array([[0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 1, 1, 1, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0]], dtype=np.float32)

def shortDiagonalMask5x5():
    return np.array([[0, 0, 0, 0, 0],
                     [0, 1, 0, 1, 0],
                     [0, 0, 1, 0, 0],
                     [0, 1, 0, 1, 0],
                     [0, 0, 0, 0, 0]], dtype=np.float32)

def tobiMask5x5():
    return np.array([[0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0]], dtype=np.float32)

def longDiagonalMask5x5():
    return np.array([[1, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 1]], dtype=np.float32)

def keimaMask5x5():
    return np.array([[0, 1, 0, 1, 0],
                     [1, 0, 0, 0, 1],
                     [0, 0, 1, 0, 0],
                     [1, 0, 0, 0, 1],
                     [0, 1, 0, 1, 0]], dtype=np.float32)


def getMask(channels, kernel_size, mask_desc):
    k_size = kernel_size[0] if isinstance(kernel_size, (tuple, list)) else kernel_size

    if k_size not in [3, 5]:
        raise ValueError("Unsupported kernel size. Only 3x3 and 5x5 are supported.")
    
    if mask_desc is None:
        mask_desc = []
    mask_desc = list(mask_desc) + [0] * (5 - len(mask_desc))
    neighbours, short_diag, tobi, long_diag, keima = mask_desc

    if k_size == 3 and any(x > 0 for x in [tobi, long_diag, keima]):
        raise ValueError("3x3 kernels do not support 'tobi', 'long diagonal', or 'keima' masks.")

    filter_list = []
    
    if k_size == 3:
        if neighbours > 0:
            filter_list.append(np.repeat(np.expand_dims(neighbourMask3x3(), axis=2), neighbours, axis=2))
        if short_diag > 0:
            filter_list.append(np.repeat(np.expand_dims(diagonalMask3x3(), axis=2), short_diag, axis=2))
        sum_masked = neighbours + short_diag
    else: 
        if neighbours > 0:
            filter_list.append(np.repeat(np.expand_dims(neighbourMask5x5(), axis=2), neighbours, axis=2))
        if short_diag > 0:
            filter_list.append(np.repeat(np.expand_dims(shortDiagonalMask5x5(), axis=2), short_diag, axis=2))
        if tobi > 0:
            filter_list.append(np.repeat(np.expand_dims(tobiMask5x5(), axis=2), tobi, axis=2))
        if long_diag > 0:
            filter_list.append(np.repeat(np.expand_dims(longDiagonalMask5x5(), axis=2), long_diag, axis=2))
        if keima > 0:
            filter_list.append(np.repeat(np.expand_dims(keimaMask5x5(), axis=2), keima, axis=2))
        sum_masked = neighbours + short_diag + tobi + long_diag + keima

    if sum_masked > channels:
        raise ValueError(f"Sum of masked channels ({sum_masked}) exceeds total input channels ({channels}).")

    remaining_channels = channels - sum_masked
    if remaining_channels > 0:
        unmasked_block = np.ones((k_size, k_size, remaining_channels), dtype=np.float32)
        filter_list.append(unmasked_block)

    m = np.concatenate(filter_list, axis=2)
    m = np.expand_dims(m, axis=3)
    
    return m


class GoMaskedDepthwise(tf.keras.layers.Conv2D):
    def __init__(self,
                 kernel_size,
                 mask_desc=None,
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
        super(GoMaskedDepthwise, self).__init__(
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
        self.precision = precision
        self.padding = self.padding.upper()
        self.mask_desc = mask_desc

    def build(self, input_shape):
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

        if self.mask_desc is not None:
            m = getMask(input_dim, self.kernel_size, self.mask_desc)
            self.mask = tf.constant(m, dtype=self.precision)
            
        self.built = True

    def call(self, inputs):
        kernel = self.depthwise_kernel
        if hasattr(self, "mask"):
            kernel = kernel * self.mask
            
        df = "NCHW" if self.data_format == 'channels_first' else "NHWC"
        strides_4d = [1, self.strides[0], self.strides[1], 1]

        outputs = tf.nn.depthwise_conv2d(
            inputs,
            kernel,
            strides=strides_4d,
            padding=self.padding,
            data_format=df
        )

        if self.bias is not None:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=df)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def get_mask(self):
        return getattr(self, "mask", None)
    
    def get_mask_desc(self):
        return getattr(self, "mask_desc", None)


def batch_norm(input_tensor, name, scale=False, axis=1):
    return tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        axis=axis,
        center=True,
        scale=scale,
        name=name)(input_tensor)


class ApplySqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, data_format='channels_first', **kwargs):
        super(ApplySqueezeExcitation, self).__init__(**kwargs)
        self.data_format = data_format

    def build(self, input_shape):
        self.reshape_size = input_shape[1][1]
        self.channels = self.reshape_size // 2

    def call(self, inputs):
        x = inputs[0]
        excited = inputs[1]
        gammas, betas = tf.split(excited, 2, axis=-1)

        if self.data_format == 'channels_first':
            gammas = tf.reshape(gammas, [-1, self.channels, 1, 1])
            betas = tf.reshape(betas, [-1, self.channels, 1, 1])
        else:
            gammas = tf.reshape(gammas, [-1, 1, 1, self.channels])
            betas = tf.reshape(betas, [-1, 1, 1, self.channels])
            
        return tf.nn.sigmoid(gammas) * x + betas


def squeeze_excitation(inputs, channels, se_ratio, name, activation,
                       data_format='channels_first'):
    assert channels % se_ratio == 0

    pooled = tf.keras.layers.GlobalAveragePooling2D(
        data_format=data_format)(inputs)
        
    squeezed = tf.keras.layers.Activation(activation)(
        tf.keras.layers.Dense(channels // se_ratio,
                              kernel_initializer='glorot_normal',
                              name=name + '_se_dense1')(pooled))
                                
    excited = tf.keras.layers.Dense(2 * channels,
                                    kernel_initializer='glorot_normal',
                                    name=name + '_se_dense2')(squeezed)

    return ApplySqueezeExcitation(data_format=data_format)([inputs, excited])


def mobilenet_block(x, dff: int, kernel_size: int, 
                    name: str, mask_desc=None, activation='mish', 
                    precision=tf.float32, se_ratio=2, data_format='channels_first'):

    channel_axis = 1 if data_format == 'channels_first' else 3
    channels = x.shape[channel_axis]
    activate = tf.keras.activations.get(activation)

    flow = tf.keras.layers.Conv2D(dff, 1,
                                  data_format=data_format,
                                  use_bias=False, 
                                  kernel_initializer='glorot_normal',
                                  name=name + "_1_pointwise")(x)
    flow = batch_norm(flow, name=name + "_1_bn", axis=channel_axis)
    flow = activate(flow)
    
    flow = GoMaskedDepthwise(kernel_size, 
                             mask_desc=mask_desc,
                             data_format=data_format,
                             padding='same',
                             use_bias=False,
                             kernel_initializer='glorot_normal',
                             name=name + "_2_depthwise",
                             precision=precision
                             )(flow)
    flow = batch_norm(flow, name=name + '_2_bn', axis=channel_axis)
    flow = activate(flow)

    flow = tf.keras.layers.Conv2D(channels, 1,
                                  data_format=data_format,
                                  use_bias=False, 
                                  kernel_initializer='glorot_normal',
                                  name=name + "_3_pointwise")(flow)
    flow = batch_norm(flow, name=name + '_3_bn', scale=True, axis=channel_axis)

    flow = squeeze_excitation(flow, channels, se_ratio, name, activation,
                              data_format=data_format)

    return tf.keras.layers.Add()([flow, x])