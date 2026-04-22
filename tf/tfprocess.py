#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import os
import random
import tensorflow as tf
import time
import bisect
import lc0_az_policy_map
import attention_policy_map as apm
import proto.net_pb2 as pb
from functools import reduce
import operator

from net import Net

def make_piece_pattern_mask(piece_type):
    # Use -10000.0 instead of -1e9 to prevent NaN overflows in mixed float16 precision
    mask = np.zeros((64, 64), dtype=float)
    for i in range(64):
        r1, c1 = divmod(i, 8)
        for j in range(64):
            r2, c2 = divmod(j, 8)
            dr = abs(r1 - r2)
            dc = abs(c1 - c2)
            
            valid = False
            # A square should always be able to attend to itself
            if i == j:
                valid = True
            elif piece_type == 'rook' and (dr == 0 or dc == 0):
                valid = True
            elif piece_type == 'bishop' and (dr == dc):
                valid = True
            elif piece_type == 'knight' and ((dr == 2 and dc == 1) or (dr == 1 and dc == 2)):
                valid = True
            elif piece_type == 'queen' and (dr == 0 or dc == 0 or dr == dc):
                valid = True
            elif piece_type == 'king' and (dr <= 1 and dc <= 1):
                valid = True
            elif piece_type == 'pawn' and ((dr == 1 and dc <= 1) or (dr == 2 and dc == 0)):
                valid = True
            elif piece_type == 'color' and ((r1 + c1) % 2 == (r2 + c2) % 2):
                valid = True
                
            if not valid:
                mask[i, j] = -10000.0
    return mask



def square_relu(x):
    return tf.nn.relu(x)**2


class Gating(tf.keras.layers.Layer):

    def __init__(self, name=None, additive=True, init_value=None, **kwargs):
        self.additive = additive
        if init_value is None:
            init_value = 0 if self.additive else 1
        self.init_value = init_value
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.gate = self.add_weight(name='gate',
                                    shape=input_shape[1:],
                                    constraint=tf.keras.constraints.NonNeg()
                                    if not self.additive else None,
                                    initializer=tf.constant_initializer(
                                        self.init_value),
                                    trainable=True)

    def call(self, inputs):
        return tf.add(inputs, self.gate) if self.additive else tf.multiply(
            inputs, self.gate)


def ma_gating(inputs, name):
    out = Gating(name=name + '/mult_gate', additive=False)(inputs)
    out = Gating(name=name + '/add_gate', additive=True)(out)
    return out



class ApplyPolicyMap(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ApplyPolicyMap, self).__init__(**kwargs)
        self.fc1 = tf.constant(lc0_az_policy_map.make_map())

    def call(self, inputs):
        h_conv_pol_flat = tf.reshape(inputs, [-1, 80 * 8 * 8])
        return tf.matmul(h_conv_pol_flat,
                         tf.cast(self.fc1, h_conv_pol_flat.dtype))


class ApplyAttentionPolicyMap(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ApplyAttentionPolicyMap, self).__init__(**kwargs)
        self.fc1 = tf.constant(apm.make_map())

    def call(self, logits, pp_logits):
        logits = tf.concat([
            tf.reshape(logits, [-1, 64 * 64]),
            tf.reshape(pp_logits, [-1, 8 * 24])
        ],
                           axis=1)
        return tf.matmul(logits, tf.cast(self.fc1, logits.dtype))


class Metric:

    def __init__(self, short_name, long_name, suffix='', **kwargs):
        self.short_name = short_name
        self.long_name = long_name
        self.suffix = suffix
        self.value = 0.0
        self.count = 0

    def assign(self, value):
        self.value = value
        self.count = 1

    def accumulate(self, value):
        if self.count > 0:
            self.value = self.value + value
            self.count = self.count + 1
        else:
            self.assign(value)

    def merge(self, other):
        assert self.short_name == other.short_name
        self.value = self.value + other.value
        self.count = self.count + other.count

    def get(self):
        if self.count == 0:
            return self.value
        return self.value / self.count

    def reset(self):
        self.value = 0.0
        self.count = 0


class TFProcess:

    def __init__(self, cfg):
        self.cfg = cfg
        self.net = Net()
        self.root_dir = os.path.join(self.cfg['training']['path'],
                                     self.cfg['name'])

        # --- PLAIN TEXT LOGGER SETUP ---
        # Get the model name from your YAML, default to 'model' if missing
        model_name = self.cfg.get('name', 'model')
        
        # Save it in the same directory as your network weights
        log_dir = self.cfg['training'].get('logs_path', './logs/')
        os.makedirs(log_dir, exist_ok=True)
        self.txt_log_file = os.path.join(log_dir, f"{model_name}.txt")

        # Write a header every time the script is started/resumed
        with open(self.txt_log_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f" RUN STARTED/RESUMED: {model_name}\n")
            f.write(f"{'='*60}\n")
        # ------------------------------

        self.encoder_layers = self.cfg['model'].get('encoder_layers', 0)
        self.encoder_heads = self.cfg['model'].get('encoder_heads', 2)
        self.embedding_size = self.cfg['model']['embedding_size']
        

        self.policy_channels = self.cfg['model'].get('policy_channels', 32)
        self.pol_embedding_size = self.cfg['model'].get(
            'pol_embedding_size', self.embedding_size)
        self.val_embedding_size = self.cfg['model'].get(
            'value_embedding_size', 32)
        self.mov_embedding_size = self.cfg['model'].get(
            'moves_left_embedding_size', 8)
        #policy head
        self.pol_encoder_layers = (0 if self.encoder_layers > 0 else 1)
        #logic is to explictly warn users who set both in yaml
        if self.cfg['model'].get('pol_encoder_layers') is not None:
            self.pol_encoder_layers = self.cfg['model'].get(
                'pol_encoder_layers')

        self.pol_encoder_heads = self.cfg['model'].get('pol_encoder_heads', 2)
        self.pol_encoder_d_model = self.cfg['model'].get(
            'pol_encoder_d_model', self.embedding_size)
        self.pol_encoder_dff = self.cfg['model'].get(
            'pol_encoder_dff', (self.embedding_size * 1.5) // 1)
        self.policy_d_model = self.cfg['model'].get('policy_d_model',
                                                    self.embedding_size)

        #encoder body
        self.input_gate = self.cfg['model'].get('input_gate')
        self.encoder_d_model = self.cfg['model'].get('encoder_d_model')
        self.encoder_dff = self.cfg['model'].get(
            'encoder_dff', (self.embedding_size * 1.5) // 1)
        self.policy_d_model = self.cfg['model'].get('policy_d_model',
                                                    self.embedding_size)
        self.arc_encoding = self.cfg['model'].get('arc_encoding', True)
        self.square_relu_ffn = self.cfg['model'].get('square_relu_ffn', False)

        self.soft_policy_temperature = self.cfg["model"].get(
            "soft_policy_temperature", 1.0)

        self.use_smolgen = self.cfg['model'].get('use_smolgen', False)
        self.smolgen_hidden_channels = self.cfg['model'].get(
            'smolgen_hidden_channels', 16)
        self.smolgen_hidden_sz = self.cfg['model'].get('smolgen_hidden_sz',
                                                       128)
        self.smolgen_gen_sz = self.cfg['model'].get('smolgen_gen_sz', 128)
        self.smolgen_activation = self.cfg['model'].get(
            'smolgen_activation', 'swish')

        self.dropout_rate = self.cfg['model'].get('dropout_rate', 0.0)
        precision = self.cfg['training'].get('precision', 'single')
        loss_scale = self.cfg['training'].get('loss_scale', 128)
        self.virtual_batch_size = self.cfg['model'].get(
            'virtual_batch_size', None)

        if precision == 'single':
            self.model_dtype = tf.float32
        elif precision == 'half':
            self.model_dtype = tf.float16
        else:
            raise ValueError("Unknown precision: {}".format(precision))

        # Scale the loss to prevent gradient underflow
        self.loss_scale = 1 if self.model_dtype == tf.float32 else loss_scale

        policy_head = self.cfg['model'].get('policy', 'attention')
        value_head = self.cfg['model'].get('value', 'wdl')
        moves_left_head = self.cfg['model'].get('moves_left', 'v1')
        input_mode = self.cfg['model'].get('input_type', 'classic')
        default_activation = self.cfg['model'].get('default_activation',
                                                   'relu')

        self.POLICY_HEAD = None
        self.VALUE_HEAD = None
        self.MOVES_LEFT_HEAD = None
        self.INPUT_MODE = None
        self.DEFAULT_ACTIVATION = None

        if policy_head == "classical":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_CLASSICAL
        elif policy_head == "convolution":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_CONVOLUTION
        elif policy_head == "attention":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_ATTENTION
            if self.pol_encoder_layers > 0:
                self.net.set_pol_headcount(self.pol_encoder_heads)
        else:
            raise ValueError(
                "Unknown policy head format: {}".format(policy_head))

        self.net.set_policyformat(self.POLICY_HEAD)

        if value_head == "classical":
            self.VALUE_HEAD = pb.NetworkFormat.VALUE_CLASSICAL
            self.wdl = False
        elif value_head == "wdl":
            self.VALUE_HEAD = pb.NetworkFormat.VALUE_WDL
            self.wdl = True
        else:
            raise ValueError(
                "Unknown value head format: {}".format(value_head))

        self.net.set_valueformat(self.VALUE_HEAD)

        if moves_left_head == "none":
            self.MOVES_LEFT_HEAD = pb.NetworkFormat.MOVES_LEFT_NONE
            self.moves_left = False
        elif moves_left_head == "v1":
            self.MOVES_LEFT_HEAD = pb.NetworkFormat.MOVES_LEFT_V1
            self.moves_left = True
        else:
            raise ValueError(
                "Unknown moves left head format: {}".format(moves_left_head))

        self.net.set_movesleftformat(self.MOVES_LEFT_HEAD)

        if input_mode == "classic":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_CLASSICAL_112_PLANE
        elif input_mode == "frc_castling":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CASTLING_PLANE
        elif input_mode == "canonical":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION
        elif input_mode == "canonical_100":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES
        elif input_mode == "canonical_armageddon":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON
        elif input_mode == "canonical_v2":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2
        elif input_mode == "canonical_v2_armageddon":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON
        else:
            raise ValueError(
                "Unknown input mode format: {}".format(input_mode))

        self.net.set_input(self.INPUT_MODE)

        self.embedding_style = self.cfg["model"].get("embedding_style", "old").lower()
        self.embedding_dense_sz = self.cfg["model"].get("embedding_dense_sz", 128)
        
        if self.embedding_style == "new":
            self.net.set_input_embedding(
                pb.NetworkFormat.INPUT_EMBEDDING_PE_DENSE)
        elif self.encoder_layers > 0:
            self.net.set_input_embedding(
                pb.NetworkFormat.INPUT_EMBEDDING_PE_MAP)
        else:
            self.net.set_input_embedding(
                pb.NetworkFormat.INPUT_EMBEDDING_NONE)

        if default_activation == "relu":
            self.net.set_defaultactivation(
                pb.NetworkFormat.DEFAULT_ACTIVATION_RELU)
            self.DEFAULT_ACTIVATION = 'relu'
        elif default_activation == "mish":
            self.net.set_defaultactivation(
                pb.NetworkFormat.DEFAULT_ACTIVATION_MISH)
            try:
                self.DEFAULT_ACTIVATION = tf.keras.activations.mish
            except AttributeError:
                import tensorflow_addons as tfa
                self.DEFAULT_ACTIVATION = tfa.activations.mish
        else:
            raise ValueError("Unknown default activation type: {}".format(
                default_activation))

        if self.encoder_layers > 0:
            self.net.set_headcount(self.encoder_heads)
            self.net.set_networkformat(
                pb.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT)
            self.net.set_smolgen_activation(
                self.net.activation(self.smolgen_activation))
            self.net.set_ffn_activation(
                self.net.activation(
                    'sqrrelu' if self.square_relu_ffn else 'default'))

        self.swa_enabled = self.cfg['training'].get('swa', False)

        # Limit momentum of SWA exponential average to 1 - 1/(swa_max_n + 1)
        self.swa_max_n = self.cfg['training'].get('swa_max_n', 0)

        self.renorm_enabled = self.cfg['training'].get('renorm', False)
        self.renorm_max_r = self.cfg['training'].get('renorm_max_r', 1)
        self.renorm_max_d = self.cfg['training'].get('renorm_max_d', 0)
        self.renorm_momentum = self.cfg['training'].get(
            'renorm_momentum', 0.99)

        if self.cfg['gpu'] == 'all':
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self.strategy = tf.distribute.MirroredStrategy()
            tf.distribute.experimental_set_strategy(self.strategy)
        else:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print(gpus)
            tf.config.experimental.set_visible_devices(gpus[self.cfg['gpu']],
                                                       'GPU')
            tf.config.experimental.set_memory_growth(gpus[self.cfg['gpu']],
                                                     True)
            self.strategy = None
        if self.model_dtype == tf.float16:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        self.global_step = tf.Variable(0,
                                       name='global_step',
                                       trainable=False,
                                       dtype=tf.int64)

        self.attention_masks_cfg = self.cfg["model"].get("attention_masks", [])
        num_layers = self.encoder_layers 
        
        mha_mask_np = np.zeros((num_layers, 1, self.encoder_heads, 64, 64), dtype=float)
        
        for rule in self.attention_masks_cfg:
            piece = rule['piece']
            heads = rule['heads']
            layers = range(num_layers) if rule['layers'] == "all" else rule['layers']
            
            piece_mask = make_piece_pattern_mask(piece)
            for l in layers:
                if l < num_layers:
                    for h in heads:
                        if h < self.encoder_heads:
                            mha_mask_np[l, 0, h, :, :] = piece_mask
                            
        self.mha_mask = tf.constant(mha_mask_np, dtype=self.model_dtype)
        # -----------------------------------------------

    def init(self, train_dataset, test_dataset, validation_dataset=None):
        if self.strategy is not None:
            self.train_dataset = self.strategy.experimental_distribute_dataset(
                train_dataset)
        else:
            self.train_dataset = train_dataset
        self.train_iter = iter(self.train_dataset)
        if self.strategy is not None:
            self.test_dataset = self.strategy.experimental_distribute_dataset(
                test_dataset)
        else:
            self.test_dataset = test_dataset
        self.test_iter = iter(self.test_dataset)
        if self.strategy is not None and validation_dataset is not None:
            self.validation_dataset = self.strategy.experimental_distribute_dataset(
                validation_dataset)
        else:
            self.validation_dataset = validation_dataset
        if self.strategy is not None:
            this = self
            with self.strategy.scope():
                this.init_net()
        else:
            self.init_net()

    def init_net(self):
        #self.l2reg = tf.keras.regularizers.l2(l=0.5 * (0.0001))
        input_var = tf.keras.Input(shape=(112, 8, 8))
        outputs = self.construct_net(input_var)
        self.model = tf.keras.Model(inputs=input_var, outputs=outputs)

        print(f"Total Params: {np.sum([np.prod(w.shape) for w in self.model.trainable_weights]):,}")
        
        try:
            import tensorflow_models as tfm
            flops = tfm.core.train_utils.try_count_flops(self.model)
            print(f"FLOPS: {flops / 10 ** 9:.03} G")
        except ImportError:
            print("To see FLOPs count, run: pip install tf-models-official")

        # swa_count initialized regardless to make checkpoint code simpler.
        self.swa_count = tf.Variable(0., name='swa_count', trainable=False)
        self.swa_weights = None
        if self.swa_enabled:
            # Count of networks accumulated into SWA
            self.swa_weights = [
                tf.Variable(w, trainable=False) for w in self.model.weights
            ]


        self.optimizer_name = self.cfg['training'].get('optimizer', 'sgd').lower()
        self.beta_1 = self.cfg['training'].get('beta_1', 0.9)
        self.beta_2 = self.cfg['training'].get('beta_2', 0.999)
        self.epsilon = self.cfg['training'].get('epsilon', 1e-7)
        #self.weight_decay = self.cfg["training"].get("weight_decay", 0.0005)
        self.weight_decay = self.cfg["training"].get("weight_decay", 0.0)
        self.active_lr = tf.Variable(0.000001, trainable=False)
        # All 'new' (TF 2.10 or newer non-legacy) optimizers must have learning_rate updated manually.
        self.update_lr_manually = True
        # Be sure not to set new_optimizer before TF 2.11, or unless you edit the code to specify a new optimizer explicitly.
        if self.optimizer_name == "sgd":
            if self.cfg['training'].get('new_optimizer'):
                self.optimizer = tf.keras.optimizers.SGD(
                    learning_rate=self.active_lr, momentum=0.9, nesterov=True)
                self.update_lr_manually = True
            else:
                try:
                    self.optimizer = tf.keras.optimizers.legacy.SGD(
                        learning_rate=lambda: self.active_lr,
                        momentum=0.9,
                        nesterov=True)
                except AttributeError:
                    self.optimizer = tf.keras.optimizers.SGD(
                        learning_rate=lambda: self.active_lr,
                        momentum=0.9,
                        nesterov=True)
        elif self.optimizer_name == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=self.active_lr, rho=0.9, momentum=0.0, epsilon=1e-07, centered=True)
        elif self.optimizer_name == "nadam":
            self.optimizer = tf.keras.optimizers.Nadam(
                learning_rate=self.active_lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon, weight_decay = self.weight_decay)
        else:
            raise ValueError("Unknown optimizer: " + self.optimizer_name)

        self.orig_optimizer = self.optimizer
        try:
            self.aggregator = self.orig_optimizer.aggregate_gradients
        except AttributeError:
            self.aggregator = self.orig_optimizer.gradient_aggregator
        if self.loss_scale != 1:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                self.optimizer, dynamic=True)

        def split_value_buckets(x, n_buckets=None, lo=-1.0, hi=1.0):
            if n_buckets is None:
                n_buckets = self.categorical_value_buckets
            x = tf.cast(x, tf.float32)
            
            epsilon = 1e-5
            x = tf.clip_by_value(x, lo, hi - epsilon)
            
            x = (x - lo) / (hi - lo) * n_buckets
            x = tf.cast(x, tf.int32)
            return tf.one_hot(x, n_buckets, dtype=tf.float32)

        def categorical_value_loss(target, output):
            target = convert_val_to_scalar(target, softmax=False)
            target = split_value_buckets(target)
            output = tf.clip_by_value(output, -20.0, 20.0)
            
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(target), logits=output)

            loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))
            
            return tf.reduce_mean(loss)


        def correct_policy(target, output, temperature=1.0):
            #output = tf.cast(output, tf.float32)
            # Calculate loss on policy head
            if self.cfg['training'].get('mask_legal_moves'):
                # extract mask for legal moves from target policy
                move_is_legal = tf.greater_equal(target, 0)
                # replace logits of illegal moves with large negative value (so that it doesn't affect policy of legal moves) without gradient
                illegal_filler = tf.zeros_like(output) - 1.0e10
                output = tf.where(move_is_legal, output, illegal_filler)
            # y_ still has -1 on illegal moves, flush them to 0
            target = tf.pow(tf.nn.relu(target), 1.0 / temperature)
            # normalize
            target = target / \
                tf.reduce_sum(input_tensor=target, axis=1, keepdims=True)
            return target, output

        def policy_loss(target, output):
            target, output = correct_policy(target, output)
            policy_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(target), logits=output)
            return tf.reduce_mean(input_tensor=policy_cross_entropy)

        self.policy_loss_fn = policy_loss

        def policy_accuracy(target, output):
            target, output = correct_policy(target, output)
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(input=target, axis=1),
                             tf.argmax(input=output, axis=1)), tf.float32))

        self.policy_accuracy_fn = policy_accuracy

        def moves_left_mean_error_fn(target, output):
            output = tf.cast(output, tf.float32)
            return tf.reduce_mean(tf.abs(target - output))

        self.moves_left_mean_error = moves_left_mean_error_fn

        def policy_entropy(target, output):
            target, output = correct_policy(target, output)
            softmaxed = tf.nn.softmax(output)
            return tf.math.negative(
                tf.reduce_mean(
                    tf.reduce_sum(tf.math.xlogy(softmaxed, softmaxed),
                                  axis=1)))

        self.policy_entropy_fn = policy_entropy

        def policy_uniform_loss(target, output):
            uniform = tf.where(tf.greater_equal(target, 0),
                               tf.ones_like(target), tf.zeros_like(target))
            balanced_uniform = uniform / tf.reduce_sum(
                uniform, axis=1, keepdims=True)
            target, output = correct_policy(target, output)
            policy_cross_entropy = \
                tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(balanced_uniform),
                                                        logits=output)
            return tf.reduce_mean(input_tensor=policy_cross_entropy)

        self.policy_uniform_loss_fn = policy_uniform_loss

        q_ratio = self.cfg['training'].get('q_ratio', 0)
        assert 0 <= q_ratio <= 1

        # Linear conversion to scalar to compute MSE with, for comparison to old values
        wdl = tf.expand_dims(tf.constant([1.0, 0.0, -1.0]), 1)

        self.qMix = lambda z, q: q * q_ratio + z * (1 - q_ratio)
        # Loss on value head
        if self.wdl:

            def value_loss(target, output):
                output = tf.cast(output, tf.float32)
                value_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.stop_gradient(target), logits=output)
                return tf.reduce_mean(input_tensor=value_cross_entropy)

            self.value_loss_fn = value_loss

            def mse_loss(target, output):
                output = tf.cast(output, tf.float32)
                scalar_z_conv = tf.matmul(tf.nn.softmax(output), wdl)
                scalar_target = tf.matmul(target, wdl)
                return tf.reduce_mean(input_tensor=tf.math.squared_difference(
                    scalar_target, scalar_z_conv))

            self.mse_loss_fn = mse_loss
        else:

            def value_loss(target, output):
                return tf.constant(0)

            self.value_loss_fn = value_loss

            def mse_loss(target, output):
                output = tf.cast(output, tf.float32)
                scalar_target = tf.matmul(target, wdl)
                return tf.reduce_mean(input_tensor=tf.math.squared_difference(
                    scalar_target, output))

            self.mse_loss_fn = mse_loss

        if self.moves_left:

            def moves_left_loss(target, output):
                # Scale the loss to similar range as other losses.
                scale = 20.0
                target = target / scale
                output = tf.cast(output, tf.float32) / scale
                if self.strategy is not None:
                    huber = tf.keras.losses.Huber(
                        10.0 / scale, reduction=tf.keras.losses.Reduction.NONE)
                else:
                    huber = tf.keras.losses.Huber(10.0 / scale)
                return tf.reduce_mean(huber(target, output))
        else:
            moves_left_loss = None

        self.moves_left_loss_fn = moves_left_loss

        pol_loss_w = self.cfg['training']['policy_loss_weight']
        val_loss_w = self.cfg['training']['value_loss_weight']

        if self.moves_left:
            moves_loss_w = self.cfg['training']['moves_left_loss_weight']
        else:
            moves_loss_w = tf.constant(0.0, dtype=tf.float32)
        reg_term_w = self.cfg['training'].get('reg_term_weight', 1.0)

        def _lossMix(policy, value, moves_left, reg_term):
            return pol_loss_w * policy + val_loss_w * value + moves_loss_w * moves_left + reg_term_w * reg_term

        self.lossMix = _lossMix

        def accuracy(target, output):
            output = tf.cast(output, tf.float32)
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(input=target, axis=1),
                             tf.argmax(input=output, axis=1)), tf.float32))

        self.accuracy_fn = accuracy

        # Order must match the order in process_inner_loop
        self.train_metrics = [
            Metric('P', 'Policy Loss'),
            Metric('V', 'Value Loss'),
            Metric('ML', 'Moves Left Loss'),
            Metric('Reg', 'Reg term'),
            Metric('Total', 'Total Loss'),
            Metric(
                'V MSE', 'MSE Loss'
            ),  # Long name here doesn't mention value for backwards compatibility reasons.
        ]
        self.time_start = None
        self.last_steps = None

        # Order must match the order in calculate_test_summaries_inner_loop
        self.test_metrics = [
            Metric('P', 'Policy Loss'),
            Metric('V', 'Value Loss'),
            Metric('ML', 'Moves Left Loss'),
            Metric('V MSE', 'MSE Loss'),  # Long name here doesn't mention value for backwards compatibility reasons.
            Metric('P Acc', 'Policy Accuracy', suffix='%'),
            Metric('V Acc', 'Value Accuracy', suffix='%'),
            Metric('ML Mean', 'Moves Left Mean Error'),
            Metric('P Entropy', 'Policy Entropy'),
            Metric('P UL', 'Policy UL'),
        ]

        # Set adaptive learning rate during training
        self.cfg['training']['lr_boundaries'].sort()
        self.warmup_steps = self.cfg['training'].get('warmup_steps', 0)
        self.lr = self.cfg['training']['lr_values'][0]
        '''
        self.test_writer = tf.summary.create_file_writer(
            os.path.join(os.getcwd(),
                         "leelalogs/{}-test".format(self.cfg['name'])))
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(os.getcwd(),
                         "leelalogs/{}-train".format(self.cfg['name'])))
        if vars(self).get('validation_dataset', None) is not None:
            self.validation_writer = tf.summary.create_file_writer(
                os.path.join(
                    os.getcwd(),
                    "leelalogs/{}-validation".format(self.cfg['name'])))
        if self.swa_enabled:
            self.swa_writer = tf.summary.create_file_writer(
                os.path.join(os.getcwd(),
                             "leelalogs/{}-swa-test".format(self.cfg['name'])))
            self.swa_validation_writer = tf.summary.create_file_writer(
                os.path.join(
                    os.getcwd(),
                    "leelalogs/{}-swa-validation".format(self.cfg['name'])))
        '''
        self.checkpoint = tf.train.Checkpoint(optimizer=self.orig_optimizer,
                                              model=self.model,
                                              global_step=self.global_step,
                                              swa_count=self.swa_count)
        self.checkpoint.listed = self.swa_weights
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.root_dir,
            max_to_keep=50,
            keep_checkpoint_every_n_hours=24,
            checkpoint_name=self.cfg['name'])

    def log_metrics_to_file(self, message):
        """Appends a string to the plain text log file."""
        with open(self.txt_log_file, "a") as f:
            f.write(message + "\n")

    def replace_weights(self, proto_filename, ignore_errors=False):
        self.net.parse_proto(proto_filename)

        filters, blocks = self.net.filters(), self.net.blocks()
        if not ignore_errors:
            if self.embedding_size != filters:
                raise ValueError("Number of filters doesn't match the network")
            if self.RESIDUAL_BLOCKS != blocks:
                raise ValueError("Number of blocks doesn't match the network")
            if self.POLICY_HEAD != self.net.pb.format.network_format.policy:
                raise ValueError("Policy head type doesn't match the network")
            if self.VALUE_HEAD != self.net.pb.format.network_format.value:
                raise ValueError("Value head type doesn't match the network")

        # List all tensor names we need weights for.
        names = []
        for weight in self.model.weights:
            names.append(weight.name)

        new_weights = self.net.get_weights_v2(names)
        for weight in self.model.weights:
            if 'renorm' in weight.name:
                # Renorm variables are not populated.
                continue

            try:
                new_weight = new_weights[weight.name]
            except KeyError:
                error_string = 'No values for tensor {} in protobuf'.format(
                    weight.name)
                if ignore_errors:
                    print(error_string)
                    continue
                else:
                    raise KeyError(error_string)

            if reduce(operator.mul, weight.shape.as_list(),
                      1) != len(new_weight):
                error_string = 'Tensor {} has wrong length. Tensorflow shape {}, size in protobuf {}'.format(
                    weight.name, weight.shape.as_list(), len(new_weight))
                if ignore_errors:
                    print(error_string)
                    continue
                else:
                    raise KeyError(error_string)

            '''
            if weight.shape.ndims == 4:
                # Rescale rule50 related weights as clients do not normalize the input.
                if weight.name == 'input/conv2d/kernel:0' and self.net.pb.format.network_format.input < pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
                    num_inputs = 112
                    # 50 move rule is the 110th input, or 109 starting from 0.
                    rule50_input = 109
                    for i in range(len(new_weight)):
                        if (i % (num_inputs * 9)) // 9 == rule50_input:
                            new_weight[i] = new_weight[i] * 99

                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weight.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weight, shape=shape)
                weight.assign(tf.transpose(a=new_weight, perm=[2, 3, 1, 0]))
            elif weight.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weight.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weight, shape=shape)
                weight.assign(tf.transpose(a=new_weight, perm=[1, 0]))
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weight, shape=weight.shape)
                weight.assign(new_weight)
            '''
            if weight.shape.ndims == 4:
                new_weight = tf.constant(new_weight, shape=weight.shape)
            elif weight.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF, [out, in] in Leela
                s = weight.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weight, shape=shape)
                new_weight = tf.transpose(a=new_weight, perm=[1, 0])
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weight, shape=weight.shape)
            
            # Modern TF2 broadcasting prevents shape mismatch crashes
            new_weight = tf.broadcast_to(new_weight, weight.shape)
            weight.assign(new_weight)
        # Replace the SWA weights as well, ensuring swa accumulation is reset.
        if self.swa_enabled:
            self.swa_count.assign(tf.constant(0.))
            self.update_swa()
        # This should result in identical file to the starting one
        # self.save_leelaz_weights('restored.pb.gz')

    def restore(self):
        if self.manager.latest_checkpoint is not None:
            print("Restoring from {0}".format(self.manager.latest_checkpoint))
            self.checkpoint.restore(self.manager.latest_checkpoint)

    '''
    def process_loop(self, batch_size, test_batches, batch_splits=1):
        if self.swa_enabled:
            # split half of test_batches between testing regular weights and SWA weights
            test_batches //= 2
        # Make sure that ghost batch norm can be applied
        if self.virtual_batch_size and batch_size % self.virtual_batch_size != 0:
            # Adjust required batch size for batch splitting.
            required_factor = self.virtual_batch_size * self.cfg[
                'training'].get('num_batch_splits', 1)
            raise ValueError(
                'batch_size must be a multiple of {}'.format(required_factor))

        # Get the initial steps value in case this is a resume from a step count
        # which is not a multiple of total_steps.
        steps = self.global_step.read_value()
        self.last_steps = steps
        self.time_start = time.time()
        self.profiling_start_step = None

        total_steps = self.cfg['training']['total_steps']
        for _ in range(steps % total_steps, total_steps):
            self.process(batch_size, test_batches, batch_splits=batch_splits)
    '''
    def process_loop(self, batch_size, test_batches, batch_splits=1):
        if self.swa_enabled:
            # split half of test_batches between testing regular weights and SWA weights
            test_batches //= 2
        # Make sure that ghost batch norm can be applied
        if self.virtual_batch_size and batch_size % self.virtual_batch_size != 0:
            # Adjust required batch size for batch splitting.
            required_factor = self.virtual_batch_size * self.cfg[
                "training"].get("num_batch_splits", 1)
            raise ValueError(
                "batch_size must be a multiple of {}".format(required_factor))

        # Get the initial steps value in case this is a resume from a step count
        # which is not a multiple of total_steps.
        steps = self.global_step.read_value()
        self.last_steps = steps
        self.time_start = time.time()
        self.profiling_start_step = None

        total_steps = self.cfg["training"]["total_steps"]

        def loop():
            for _ in range(steps % total_steps, total_steps):
                while os.path.exists("stop"):
                    time.sleep(1)
                self.process(batch_size, test_batches,
                             batch_splits=batch_splits)

        from importlib.util import find_spec
        if find_spec("rich") is not None:
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
            from rich.table import Column

            self.progressbar = Progress(
                BarColumn(),
                "[progress.percentage]{task.percentage:>4.2f}%",
                TimeRemainingColumn(),
                TextColumn("{task.completed:.2f} of {task.total} steps completed.",
                           table_column=Column(ratio=1)),
                SpinnerColumn(),
            )
            with self.progressbar:
                self.progresstask = self.progressbar.add_task(
                    f"[green]Doing {total_steps} training steps", total=total_steps)
                try:
                    loop()
                except tf.errors.ResourceExhaustedError as e:
                    print("\n" + "="*50)
                    print(">>> FATAL OOM ERROR CAUGHT <<<")
                    print(e)
                    print("="*50 + "\n")
                    print("Memory resources exhausted. Try decreasing batch size or model dimension")
                    print("Saving model...")
                    
                    # Optional: Automatically save checkpoint on OOM
                    evaled_steps = self.global_step.read_value().numpy()
                    self.manager.save(checkpoint_number=evaled_steps)
                    print(f"Emergency Model saved in file: {self.manager.latest_checkpoint}")
                    exit()
        else:
            print("Warning: 'rich' module not found, disabling progress bar.")
            loop()

    @tf.function()
    def read_weights(self):
        return [w.read_value() for w in self.model.weights]

    @tf.function()
    def process_inner_loop(self, x, y, z, q, m):
        with tf.GradientTape() as tape:
            outputs = self.model(x, training=True)
            policy = outputs[0]
            value = outputs[1]
            policy_loss = self.policy_loss_fn(y, policy)
            reg_term = sum(self.model.losses)
            if self.wdl:
                value_ce_loss = self.value_loss_fn(self.qMix(z, q), value)
                value_loss = value_ce_loss
            else:
                value_mse_loss = self.mse_loss_fn(self.qMix(z, q), value)
                value_loss = value_mse_loss
            if self.moves_left:
                moves_left = outputs[2]
                moves_left_loss = self.moves_left_loss_fn(m, moves_left)
            else:
                moves_left_loss = tf.constant(0.)
            total_loss = self.lossMix(policy_loss, value_loss, moves_left_loss,
                                      reg_term)
            if self.loss_scale != 1:
                total_loss = self.optimizer.get_scaled_loss(total_loss)
        if self.wdl:
            mse_loss = self.mse_loss_fn(self.qMix(z, q), value)
        else:
            value_loss = self.value_loss_fn(self.qMix(z, q), value)
        metrics = [
            policy_loss,
            value_loss,
            moves_left_loss,
            reg_term,
            total_loss,
            # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
            # get comparable values.
            mse_loss / 4.0,
        ]
        return metrics, tape.gradient(total_loss, self.model.trainable_weights)

    @tf.function()
    def strategy_process_inner_loop(self, x, y, z, q, m):
        metrics, new_grads = self.strategy.run(self.process_inner_loop,
                                               args=(x, y, z, q, m))
        metrics = [
            self.strategy.reduce(tf.distribute.ReduceOp.MEAN, m, axis=None)
            for m in metrics
        ]
        return metrics, new_grads

    @tf.function()
    def apply_grads(self, grads, effective_batch_splits):
        grads = [
            g[0]
            for g in self.aggregator(zip(grads, self.model.trainable_weights))
        ]
        if self.loss_scale != 1:
            grads = self.optimizer.get_unscaled_gradients(grads)
        max_grad_norm = self.cfg['training'].get(
            'max_grad_norm', 10000.0) * effective_batch_splits
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_weights),
                                       experimental_aggregate_gradients=False)
        return grad_norm

    @tf.function()
    def strategy_apply_grads(self, grads, effective_batch_splits):
        grad_norm = self.strategy.run(self.apply_grads,
                                      args=(grads, effective_batch_splits))
        grad_norm = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                         grad_norm,
                                         axis=None)
        return grad_norm

    @tf.function()
    def merge_grads(self, grads, new_grads):
        return [tf.math.add(a, b) for (a, b) in zip(grads, new_grads)]

    @tf.function()
    def strategy_merge_grads(self, grads, new_grads):
        return self.strategy.run(self.merge_grads, args=(grads, new_grads))

    def train_step(self, steps, batch_size, batch_splits):
        # need to add 1 to steps because steps will be incremented after gradient update
        if (steps +
                1) % self.cfg['training']['train_avg_report_steps'] == 0 or (
                    steps + 1) % self.cfg['training']['total_steps'] == 0:
            before_weights = self.read_weights()

        # Run training for this batch
        grads = None
        for batch_id in range(batch_splits):
            x, y, z, q, m = next(self.train_iter)
            if self.strategy is not None:
                metrics, new_grads = self.strategy_process_inner_loop(
                    x, y, z, q, m)
            else:
                metrics, new_grads = self.process_inner_loop(x, y, z, q, m)
            if not grads:
                grads = new_grads
            else:
                if self.strategy is not None:
                    grads = self.strategy_merge_grads(grads, new_grads)
                else:
                    grads = self.merge_grads(grads, new_grads)
            # Keep running averages
            for acc, val in zip(self.train_metrics, metrics):
                acc.accumulate(val)

            if hasattr(self, "progressbar"):
                self.progressbar.update(self.progresstask, completed=steps.numpy(
                ).item() - 1 + (batch_id+1) / batch_splits)
        # Gradients of batch splits are summed, not averaged like usual, so need to scale lr accordingly to correct for this.
        effective_batch_splits = batch_splits
        if self.strategy is not None:
            effective_batch_splits = batch_splits * self.strategy.num_replicas_in_sync
        self.active_lr.assign(self.lr / effective_batch_splits)
        #if self.update_lr_manually:
        #    self.orig_optimizer.learning_rate = self.active_lr
        if self.strategy is not None:
            grad_norm = self.strategy_apply_grads(grads,
                                                  effective_batch_splits)
        else:
            grad_norm = self.apply_grads(grads, effective_batch_splits)

        # Note: grads variable at this point has not been unscaled or
        # had clipping applied. Since no code after this point depends
        # upon that it seems fine for now.

        # Update steps.
        self.global_step.assign_add(1)
        steps = self.global_step.read_value()

        if steps % self.cfg['training'][
                'train_avg_report_steps'] == 0 or steps % self.cfg['training'][
                    'total_steps'] == 0:
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                steps_elapsed = steps - self.last_steps
                speed = batch_size * (tf.cast(steps_elapsed, tf.float32) /
                                      elapsed)

            print("\n")
            print("-"*60)
            print("Train step {}, lr={:g}".format(steps, self.lr), end="\n")
            self.log_metrics_to_file(f"\nTrain step {steps} : ")
            for metric in self.train_metrics:
                print(" > {} = {:g}{}".format(metric.long_name, metric.get(),
                                        metric.suffix), end="\n")

                self.log_metrics_to_file(f" > {metric.long_name} = {metric.get()}{metric.suffix}")

            print(" > ({:g} pos/s)".format(speed))


            after_weights = self.read_weights()
            '''
            with self.train_writer.as_default():
                for metric in self.train_metrics:
                    tf.summary.scalar(metric.long_name,
                                      metric.get(),
                                      step=steps)
                tf.summary.scalar("LR", self.lr, step=steps)
                tf.summary.scalar("Gradient norm",
                                  grad_norm / effective_batch_splits,
                                  step=steps)
                self.compute_update_ratio(before_weights, after_weights, steps)
            self.train_writer.flush()
            '''

            self.time_start = time_end
            self.last_steps = steps
            for metric in self.train_metrics:
                metric.reset()
        return steps

    def process(self, batch_size, test_batches, batch_splits):
        # Get the initial steps value before we do a training step.
        steps = self.global_step.read_value()

        # By default disabled since 0 != 10.
        if steps % self.cfg['training'].get('profile_step_freq',
                                            1) == self.cfg['training'].get(
                                                'profile_step_offset', 10):
            self.profiling_start_step = steps
            tf.profiler.experimental.start(
                os.path.join(os.getcwd(),
                             "leelalogs/{}-profile".format(self.cfg['name'])))

        # Run test before first step to see delta since end of last run.
        if steps % self.cfg['training']['total_steps'] == 0:
            with tf.profiler.experimental.Trace("Test", step_num=steps + 1):
                # Steps is given as one higher than current in order to avoid it
                # being equal to the value the end of a run is stored against.
                self.calculate_test_summaries(test_batches, steps + 1)
                if self.swa_enabled:
                    self.calculate_swa_summaries(test_batches, steps + 1)

        # Determine learning rate
        lr_values = self.cfg['training']['lr_values']
        lr_boundaries = self.cfg['training']['lr_boundaries']
        steps_total = steps % self.cfg['training']['total_steps']
        self.lr = lr_values[bisect.bisect_right(lr_boundaries, steps_total)]
        if self.warmup_steps > 0 and steps < self.warmup_steps:
            self.lr = self.lr * tf.cast(steps + 1,
                                        tf.float32) / self.warmup_steps

        with tf.profiler.experimental.Trace("Train", step_num=steps):
            steps = self.train_step(steps, batch_size, batch_splits)

        if self.swa_enabled and steps % self.cfg['training']['swa_steps'] == 0:
            self.update_swa()

        # Calculate test values every 'test_steps', but also ensure there is
        # one at the final step so the delta to the first step can be calculated.
        if steps % self.cfg['training']['test_steps'] == 0 or steps % self.cfg[
                'training']['total_steps'] == 0:
            with tf.profiler.experimental.Trace("Test", step_num=steps):
                self.calculate_test_summaries(test_batches, steps)
                if self.swa_enabled:
                    self.calculate_swa_summaries(test_batches, steps)

        if self.validation_dataset is not None and (
                steps % self.cfg['training']['validation_steps'] == 0
                or steps % self.cfg['training']['total_steps'] == 0):
            with tf.profiler.experimental.Trace("Validate", step_num=steps):
                if self.swa_enabled:
                    self.calculate_swa_validations(steps)
                else:
                    self.calculate_test_validations(steps)

        # Save session and weights at end, and also optionally every 'checkpoint_steps'.
        if steps % self.cfg['training']['total_steps'] == 0 or (
                'checkpoint_steps' in self.cfg['training']
                and steps % self.cfg['training']['checkpoint_steps'] == 0):
            evaled_steps = steps.numpy()
            self.manager.save(checkpoint_number=evaled_steps)
            print("Model saved in file: {}".format(
                self.manager.latest_checkpoint))
            path = os.path.join(self.root_dir, self.cfg['name'])
            leela_path = path + "-" + str(evaled_steps)
            swa_path = path + "-swa-" + str(evaled_steps)
            self.net.pb.training_params.training_steps = evaled_steps
            self.save_leelaz_weights(leela_path)
            if self.swa_enabled:
                self.save_swa_weights(swa_path)

        if self.profiling_start_step is not None and (
                steps >= self.profiling_start_step +
                self.cfg['training'].get('profile_step_count', 0)
                or steps % self.cfg['training']['total_steps'] == 0):
            tf.profiler.experimental.stop()
            self.profiling_start_step = None

    def calculate_swa_summaries(self, test_batches, steps):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        #true_test_writer, self.test_writer = self.test_writer, self.swa_writer
        print('swa', end=' ')
        self.calculate_test_summaries(test_batches, steps)
        #self.test_writer = true_test_writer
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    @tf.function()
    def calculate_test_summaries_inner_loop(self, x, y, z, q, m):
        outputs = self.model(x, training=False)
        policy = outputs[0]
        value = outputs[1]
        policy_loss = self.policy_loss_fn(y, policy)
        policy_accuracy = self.policy_accuracy_fn(y, policy)
        policy_entropy = self.policy_entropy_fn(y, policy)
        policy_ul = self.policy_uniform_loss_fn(y, policy)
        if self.wdl:
            value_loss = self.value_loss_fn(self.qMix(z, q), value)
            mse_loss = self.mse_loss_fn(self.qMix(z, q), value)
            value_accuracy = self.accuracy_fn(self.qMix(z, q), value)
        else:
            value_loss = self.value_loss_fn(self.qMix(z, q), value)
            mse_loss = self.mse_loss_fn(self.qMix(z, q), value)
            value_accuracy = tf.constant(0.)
        if self.moves_left:
            moves_left = outputs[2]
            moves_left_loss = self.moves_left_loss_fn(m, moves_left)
            moves_left_mean_error = self.moves_left_mean_error(m, moves_left)
        else:
            moves_left_loss = tf.constant(0.)
            moves_left_mean_error = tf.constant(0.)
        metrics = [
            policy_loss,
            value_loss,
            moves_left_loss,
            mse_loss / 4,
            policy_accuracy * 100,
            value_accuracy * 100,
            moves_left_mean_error,
            policy_entropy,
            policy_ul,
        ]
        return metrics

    @tf.function()
    def strategy_calculate_test_summaries_inner_loop(self, x, y, z, q, m):
        metrics = self.strategy.run(self.calculate_test_summaries_inner_loop,
                                    args=(x, y, z, q, m))
        metrics = [
            self.strategy.reduce(tf.distribute.ReduceOp.MEAN, m, axis=None)
            for m in metrics
        ]
        return metrics

    def calculate_test_summaries(self, test_batches, steps):
        for metric in self.test_metrics:
            metric.reset()
        for _ in range(0, test_batches):
            x, y, z, q, m = next(self.test_iter)
            if self.strategy is not None:
                metrics = self.strategy_calculate_test_summaries_inner_loop(
                    x, y, z, q, m)
            else:
                metrics = self.calculate_test_summaries_inner_loop(
                    x, y, z, q, m)
            for acc, val in zip(self.test_metrics, metrics):
                acc.accumulate(val)
        self.net.pb.training_params.learning_rate = self.lr
        self.net.pb.training_params.mse_loss = self.test_metrics[3].get()
        self.net.pb.training_params.policy_loss = self.test_metrics[0].get()
        # TODO store value and value accuracy in pb
        self.net.pb.training_params.accuracy = self.test_metrics[4].get()
        '''
        with self.test_writer.as_default():
            for metric in self.test_metrics:
                tf.summary.scalar(metric.long_name, metric.get(), step=steps)
            for w in self.model.weights:
                tf.summary.histogram(w.name, w, step=steps)
        self.test_writer.flush()
        '''

        print("\n")
        print("-"*60)
        print("Test step {}".format(steps), end="\n")
        self.log_metrics_to_file(f"\nTest step {steps} :")
        for metric in self.test_metrics:
            print(" > {} = {:g}{}".format(metric.long_name, metric.get(),
                                      metric.suffix),
                  end="\n")

            self.log_metrics_to_file(f" > {metric.long_name} = {metric.get()}{metric.suffix}")


    def calculate_swa_validations(self, steps):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        #true_validation_writer, self.validation_writer = self.validation_writer, self.swa_validation_writer
        print('swa', end=' ')
        self.calculate_test_validations(steps)
        #self.validation_writer = true_validation_writer
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    def calculate_test_validations(self, steps):
        for metric in self.test_metrics:
            metric.reset()
        for (x, y, z, q, m) in self.validation_dataset:
            if self.strategy is not None:
                metrics = self.strategy_calculate_test_summaries_inner_loop(
                    x, y, z, q, m)
            else:
                metrics = self.calculate_test_summaries_inner_loop(
                    x, y, z, q, m)
            for acc, val in zip(self.test_metrics, metrics):
                acc.accumulate(val)
        '''
        with self.validation_writer.as_default():
            for metric in self.test_metrics:
                tf.summary.scalar(metric.long_name, metric.get(), step=steps)
        self.validation_writer.flush()
        '''

        print("step {}, validation:".format(steps), end='')
        for metric in self.test_metrics:
            print(" {}={:g}{}".format(metric.short_name, metric.get(),
                                      metric.suffix),
                  end='')
        print()

    @tf.function()
    def compute_update_ratio(self, before_weights, after_weights, steps):
        """Compute the ratio of gradient norm to weight norm.

        Adapted from https://github.com/tensorflow/minigo/blob/c923cd5b11f7d417c9541ad61414bf175a84dc31/dual_net.py#L567
        """
        deltas = [
            after - before
            for after, before in zip(after_weights, before_weights)
        ]
        delta_norms = [tf.math.reduce_euclidean_norm(d) for d in deltas]
        weight_norms = [
            tf.math.reduce_euclidean_norm(w) for w in before_weights
        ]
        ratios = [(tensor.name, tf.cond(w != 0., lambda: d / w, lambda: -1.))
                  for d, w, tensor in zip(delta_norms, weight_norms,
                                          self.model.weights)
                  if not 'moving' in tensor.name]
        for name, ratio in ratios:
            tf.summary.scalar('update_ratios/' + name, ratio, step=steps)
        # Filtering is hard, so just push infinities/NaNs to an unreasonably large value.
        ratios = [
            tf.cond(r > 0, lambda: tf.math.log(r) / 2.30258509299,
                    lambda: 200.) for (_, r) in ratios
        ]
        tf.summary.histogram('update_ratios_log10',
                             tf.stack(ratios),
                             buckets=1000,
                             step=steps)

    def update_swa(self):
        num = self.swa_count.read_value()
        for (w, swa) in zip(self.model.weights, self.swa_weights):
            swa.assign(swa.read_value() * (num / (num + 1.)) + w.read_value() *
                       (1. / (num + 1.)))
        self.swa_count.assign(min(num + 1., self.swa_max_n))

    def save_swa_weights(self, filename):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        self.save_leelaz_weights(filename)
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    def save_leelaz_weights(self, filename):
        numpy_weights = []
        for weight in self.model.weights:
            numpy_weights.append([weight.name, weight.numpy()])
        self.net.fill_net_v2(numpy_weights)
        if hasattr(self, 'attention_masks_cfg'):
            self.net.set_attention_masks(self.attention_masks_cfg)
        self.net.save_proto(filename)


    @staticmethod
    def split_heads(inputs, batch_size: int, num_heads: int, depth: int):
        if num_heads < 2:
            return inputs
        reshaped = tf.reshape(inputs, (batch_size, 64, num_heads, depth))
        # (batch_size, num_heads, 64, depth)
        return tf.transpose(reshaped, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self,
                                     q,
                                     k,
                                     v,
                                     name: str = None,
                                     inputs=None,
                                     layer_idx = 0):

        # 0 h 64 d, 0 h d 64
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], self.model_dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        heads = scaled_attention_logits.shape[1]

        if hasattr(self, 'mha_mask'):
            # Slice the mask for this specific layer
            layer_mask = self.mha_mask[layer_idx, :, :heads, :, :]
            scaled_attention_logits = scaled_attention_logits + layer_mask

        if self.use_smolgen:
            smolgen_weights = self.smolgen_weights(
                inputs,
                heads,
                self.smolgen_hidden_channels,
                self.smolgen_hidden_sz,
                self.smolgen_gen_sz,
                name=name + '/smolgen',
                activation=self.smolgen_activation)
            scaled_attention_logits = scaled_attention_logits + smolgen_weights

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, scaled_attention_logits

    # multi-head attention in encoder blocks

    def mha(self, inputs, emb_size: int, d_model: int, num_heads: int,
            initializer, name: str, layer_idx = 0):
        assert d_model % num_heads == 0
        depth = d_model // num_heads
        # query, key, and value vectors for self-attention
        # inputs b, 64, sz
        q = tf.keras.layers.Dense(d_model,
                                  name=name + '/wq',
                                  kernel_initializer='glorot_normal')(inputs)
        k = tf.keras.layers.Dense(d_model,
                                  name=name + '/wk',
                                  kernel_initializer='glorot_normal')(inputs)
        v = tf.keras.layers.Dense(d_model,
                                  name=name + '/wv',
                                  kernel_initializer=initializer)(inputs)

        # split q, k and v into smaller vectors of size 'depth' -- one for each head in multi-head attention
        batch_size = tf.shape(q)[0]
        q = self.split_heads(q, batch_size, num_heads, depth)
        k = self.split_heads(k, batch_size, num_heads, depth)
        v = self.split_heads(v, batch_size, num_heads, depth)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, name=name, inputs=inputs, layer_idx = layer_idx)
        if num_heads > 1:
            scaled_attention = tf.transpose(scaled_attention,
                                            perm=[0, 2, 1, 3])
            scaled_attention = tf.reshape(
                scaled_attention,
                (batch_size, -1, d_model))  # concatenate heads

        # final dense layer
        output = tf.keras.layers.Dense(
            emb_size, name=name + "/dense",
            kernel_initializer=initializer)(scaled_attention)
        return output, attention_weights

    # 2-layer dense feed-forward network in encoder blocks
    def ffn(self, inputs, emb_size: int, dff: int, initializer, name: str):
        if self.encoder_layers > 0:
            activation = square_relu if self.square_relu_ffn else tf.keras.activations.get(
                self.DEFAULT_ACTIVATION)
        else:
            activation = "selu"
        dense1 = tf.keras.layers.Dense(dff,
                                       name=name + "/dense1",
                                       kernel_initializer=initializer,
                                       activation=activation)(inputs)
        out = tf.keras.layers.Dense(emb_size,
                                    name=name + "/dense2",
                                    kernel_initializer=initializer)(dense1)
        return out


   def encoder_layer(self, inputs, emb_size: int, d_model: int, num_heads: int, dff: int, name: str, training: bool, layer_idx = 0):
        # DeepNorm
        alpha = tf.cast(tf.math.pow(
            2. * self.encoder_layers, -0.25), self.model_dtype)
        beta = tf.cast(tf.math.pow(
            8. * self.encoder_layers, -0.25), self.model_dtype)

        xavier_norm = tf.keras.initializers.VarianceScaling(
            scale=beta, mode="fan_avg", distribution="truncated_normal", seed=42)

        # multihead attention
        attn_output, attn_wts = self.mha(
            inputs, emb_size, d_model, num_heads, xavier_norm, name=name + "/mha", layer_idx=layer_idx)

        # dropout for weight regularization
        attn_output = tf.keras.layers.Dropout(
            self.dropout_rate, name=name + "/dropout1")(attn_output, training=training)

        # skip connection + layernorm
        out1 = self.encoder_norm(
            name=name+"/ln1")(inputs + attn_output * alpha)

        # feed-forward network
        ffn_output = self.ffn(out1, emb_size, dff,
                              xavier_norm, name=name + "/ffn")
        ffn_output = tf.keras.layers.Dropout(
            self.dropout_rate, name=name + "/dropout2")(ffn_output, training=training)

        out2 = self.encoder_norm(
            name=name+"/ln2")(out1 + ffn_output * alpha)

        return out2, attn_wts

    def smolgen_weights(self, inputs, heads: int, hidden_channels: int, hidden_sz: int, gen_sz: int, name: str, activation="swish"):
        compressed = tf.keras.layers.Dense(
            hidden_channels, name=name+"/compress", use_bias=False)(inputs)
        compressed = tf.reshape(compressed, [-1, 64 * hidden_channels])
        hidden = tf.keras.layers.Dense(
            hidden_sz, name=name+"/hidden1_dense", activation=activation)(compressed)

        hidden = tf.keras.layers.LayerNormalization(
            name=name+"/hidden1_ln")(hidden)
        gen_from = tf.keras.layers.Dense(
            heads * gen_sz, name=name+"/gen_from", activation=activation)(hidden)
        gen_from = tf.keras.layers.LayerNormalization(
            name=name+"/gen_from_ln", center=True)(gen_from)
        gen_from = tf.reshape(gen_from, [-1, heads, gen_sz])

        out = self.smol_weight_gen_dense(gen_from)
        return tf.reshape(out, [-1, heads, 64, 64])

    def construct_net(self, inputs, name: str = ""):
        # Policy head
        assert self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION
        # TODO: re-add support for policy encoder blocks
        # do some input processing
        if self.use_smolgen:
            self.smol_weight_gen_dense = tf.keras.layers.Dense(
                64 * 64, name=name+"smol_weight_gen", use_bias=False)
        

        if self.embedding_style == "new":
            inputs = tf.cast(inputs, self.model_dtype)
            flow = tf.transpose(inputs, perm=[0, 2, 3, 1])
            flow = tf.reshape(flow, [-1, 64, tf.shape(inputs)[1]])

            pos_info = flow[..., :12]
            pos_info_flat = tf.reshape(pos_info, [-1, 64 * 12])

            pos_info_processed = tf.keras.layers.Dense(
                64*self.embedding_dense_sz, name=name+"embedding/preprocess")(pos_info_flat)
            pos_info = tf.reshape(pos_info_processed,
                                  [-1, 64, self.embedding_dense_sz])
            flow = tf.keras.layers.Concatenate()([flow, pos_info])

            # square embedding
            flow = tf.keras.layers.Dense(self.embedding_size, kernel_initializer="glorot_normal",
                                         activation=self.DEFAULT_ACTIVATION,
                                         name=name+"embedding")(flow)
            flow = self.encoder_norm(
                name=name+"embedding/ln")(flow)
            flow = ma_gating(flow, name=name+'embedding')

            # DeepNorm
            alpha = tf.cast(tf.math.pow(
                2. * self.encoder_layers, -0.25), self.model_dtype)
            beta = tf.cast(tf.math.pow(
                8. * self.encoder_layers, -0.25), self.model_dtype)



            xavier_norm = tf.keras.initializers.VarianceScaling(
                scale=beta, mode="fan_avg", distribution="truncated_normal", seed=42)

            # feed-forward network
            ffn_output = self.ffn(flow, self.embedding_size, self.encoder_dff,
                                  xavier_norm, name=name + "embedding/ffn")


            flow = self.encoder_norm(
                name=name+"embedding/ffn_ln")(flow + ffn_output * alpha)

        elif self.embedding_style == "old":
            flow = tf.transpose(inputs, perm=[0, 2, 3, 1])
            flow = tf.reshape(flow, [-1, 64, tf.shape(inputs)[1]])

            # square embedding
            flow = tf.keras.layers.Dense(self.embedding_size,
                                         kernel_initializer='glorot_normal',
                                         activation=self.DEFAULT_ACTIVATION,
                                         name='embedding')(flow)
                
            flow = ma_gating(flow, name='embedding')

        else:
            raise ValueError(
                "Unknown embedding style: {}".format(self.embedding_style))

        attn_wts = []
        for i in range(self.encoder_layers):
            flow, attn_wts_l= self.encoder_layer(flow, self.embedding_size, self.encoder_d_model,
                                                  self.encoder_heads, self.encoder_dff,
                                                  name=name+"encoder_{}".format(i + 1), training=True, layer_idx = i)

            attn_wts.append(attn_wts_l)


        flow_ = flow

        policy_tokens = tf.keras.layers.Dense(self.pol_embedding_size, kernel_initializer="glorot_normal",
                                              activation=self.DEFAULT_ACTIVATION,
                                              name=name+"policy/embedding")(flow_)
    

        def policy_head(name, activation=None, depth=None, opponent=False):
            if depth is None:
                depth = self.policy_d_model

            # reverse the tokens along the square (second) dimension to get the opponent's perspective
            tokens = tf.reverse(policy_tokens, axis=[
                1]) if opponent else policy_tokens

            # create queries and keys for policy self-attention
            queries = tf.keras.layers.Dense(depth, kernel_initializer="glorot_normal",
                                            name=name+"/attention/wq")(tokens)
            keys = tf.keras.layers.Dense(depth, kernel_initializer="glorot_normal",
                                         name=name+"/attention/wk")(tokens)

            # POLICY SELF-ATTENTION: self-attention weights are interpreted as from->to policy
            # Bx64x64 (from 64 queries, 64 keys)
            matmul_qk = tf.matmul(queries, keys, transpose_b=True)
            # queries = tf.keras.layers.Dense(self.policy_d_model, kernel_initializer="glorot_normal",
            #                                 name="policy/attention/wq")(flow)
            # keys = tf.keras.layers.Dense(self.policy_d_model, kernel_initializer="glorot_normal",
            #                              name="policy/attention/wk")(flow)

            # PAWN PROMOTION: create promotion logits using scalar offsets generated from the promotion-  keys
            # constant for scaling
            dk = tf.math.sqrt(tf.cast(tf.shape(keys)[-1], self.model_dtype))
            promotion_keys = keys[:, -8:, :]
            # queen, rook, bishop, knight order
            promotion_offsets = tf.keras.layers.Dense(4, kernel_initializer="glorot_normal",
                                                      name=name+"/attention/ppo", use_bias=False)(promotion_keys)
            promotion_offsets = tf.transpose(
                promotion_offsets, perm=[0, 2, 1]) * dk  # Bx4x8
            # knight offset is added to the other three
            promotion_offsets = promotion_offsets[:,
                                                  :3, :] + promotion_offsets[:, 3:4, :]

            # q, r, and b promotions are offset from the default promotion logit (knight)
            # default traversals from penultimate rank to promotion rank
            n_promo_logits = matmul_qk[:, -16:-8, -8:]
            q_promo_logits = tf.expand_dims(
                n_promo_logits + promotion_offsets[:, 0:1, :], axis=3)  # Bx8x8x1
            r_promo_logits = tf.expand_dims(
                n_promo_logits + promotion_offsets[:, 1:2, :], axis=3)
            b_promo_logits = tf.expand_dims(
                n_promo_logits + promotion_offsets[:, 2:3, :], axis=3)
            promotion_logits = tf.concat(
                [q_promo_logits, r_promo_logits, b_promo_logits], axis=3)  # Bx8x8x3
            # logits now alternate a7a8q,a7a8r,a7a8b,...,
            promotion_logits = tf.reshape(promotion_logits, [-1, 8, 24])

            # scale the logits by dividing them by sqrt(d_model) to stabilize gradients
            # Bx8x24 (8 from-squares, 3x8 promotions)
            promotion_logits = promotion_logits / dk
            # Bx64x64 (64 from-squares, 64 to-squares)
            policy_attn_logits = matmul_qk / dk

            attn_wts.append(promotion_logits)
            attn_wts.append(policy_attn_logits)

            # APPLY POLICY MAP: output becomes Bx1856
            h_fc1 = ApplyAttentionPolicyMap(
                name=name+"/attention_map")(policy_attn_logits, promotion_logits)

            if activation is not None:
                h_fc1 = tf.keras.layers.Activation(activation)(h_fc1)

            # Value head
            assert self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION and self.encoder_layers > 0

            return h_fc1

        aux_depth = self.cfg['model'].get('policy_d_aux', self.policy_d_model)

        policy = policy_head(name="policy/vanilla")

        policy_optimistic_st = policy_head(
            name="policy/optimistic_st") if self.cfg['model'].get('policy_optimistic_st', False) else None

        policy_soft = policy_head(
            name="policy/soft", depth=aux_depth) if self.cfg['model'].get('soft_policy', False) else None

        
        def value_head(name, wdl=True, use_err=True, use_cat=True):
            embedded_val = tf.keras.layers.Dense(self.val_embedding_size, kernel_initializer="glorot_normal",
                                                 activation=self.DEFAULT_ACTIVATION,
                                                 name=name+"/embedding")(flow)

            h_val_flat = tf.keras.layers.Flatten()(embedded_val)
            h_fc2 = tf.keras.layers.Dense(128,
                                          kernel_initializer="glorot_normal",
                                          activation=self.DEFAULT_ACTIVATION,
                                          name=name+"/dense1")(h_val_flat)

            # WDL head
            if wdl:
                value = tf.keras.layers.Dense(3,
                                              kernel_initializer="glorot_normal",
                                              name=name+"/dense2",
                                              dtype="float32")(h_fc2)
            else:
                value = tf.keras.layers.Dense(1,
                                              kernel_initializer="glorot_normal",
                                              activation="tanh",
                                              name=name+"/dense2",
                                              dtype="float32")(h_fc2)

            if use_err:
                value_err = tf.keras.layers.Dense(
                    1, kernel_initializer="glorot_normal", name=name+"/dense_error", activation="sigmoid",
                    dtype="float32")(h_fc2)
            else:
                value_err = None

            if use_cat and self.categorical_value_buckets:
                value_cat = tf.keras.layers.Dense(
                    self.categorical_value_buckets, kernel_initializer="glorot_normal", name=name+"/dense_cat",
                    dtype="float32")(h_fc2)
            else:
                value_cat = None

            return value, value_err, value_cat
        

        value_winner, value_winner_err, value_winner_cat = value_head(
            name="value/winner", wdl=self.wdl, use_err=False, use_cat=False)
        value_q, value_q_err, value_q_cat = value_head(
            name="value/q", wdl=False, use_err=True) if self.cfg['model'].get('value_q', False) else (None, None, None)
        value_st, value_st_err, value_st_cat = value_head(
            name="value/st", wdl=False, use_err=True) if self.cfg['model'].get('value_st', False) else (None, None, None)

        # Moves left head
        if self.moves_left:
            embedded_mov = tf.keras.layers.Dense(self.mov_embedding_size, kernel_initializer="glorot_normal",
                                                 activation=self.DEFAULT_ACTIVATION,
                                                 name=name+"moves_left/embedding")(flow)

            h_mov_flat = tf.keras.layers.Flatten()(embedded_mov)

            h_fc4 = tf.keras.layers.Dense(
                128,
                kernel_initializer="glorot_normal",
                activation=self.DEFAULT_ACTIVATION,
                name=name+"moves_left/dense1")(h_mov_flat)
        
            moves_left = tf.keras.layers.Dense(1,
                                               kernel_initializer="glorot_normal",
                                               activation="relu",
                                               name=name+"moves_left/dense2",
                                               dtype="float32")(h_fc4)
            
        else:
            moves_left = None

        # attention weights added as optional output for analysis -- ignored by backend
        outputs = {
            "policy": policy,
            "policy_optimistic_st": policy_optimistic_st,
            "policy_soft": policy_soft,
            "value_winner": value_winner,
            "value_q": value_q,
            "value_q_err": value_q_err,
            "value_q_cat": value_q_cat,
            "value_st": value_st,
            "value_st_err": value_st_err,
            "value_st_cat": value_st_cat,
            "moves_left": moves_left,
        }

        if self.return_attn_wts:
            outputs["attn_wts"] = attn_wts
 
        # Tensorflow does not accept None values in the output dictionary
        none_keys = []
        for key in outputs:
            if outputs[key] is None:
                none_keys.append(key)

        for key in none_keys:
            del outputs[key]

        for key in outputs:
            try:
                outputs[key] = tf.cast(outputs[key], tf.float32)
            except:
                assert key == "attn_wts"
        return outputs








    '''
    def encoder_layer(self, inputs, emb_size: int, d_model: int,
                      num_heads: int, dff: int, name: str, layer_idx = 0):
        initializer = None
        if self.encoder_layers > 0:
            # DeepNorm
            alpha = tf.cast(tf.math.pow(2. * self.encoder_layers, 0.25),
                            self.model_dtype)
            beta = tf.cast(tf.math.pow(8. * self.encoder_layers, -0.25),
                           self.model_dtype)
            xavier_norm = tf.keras.initializers.VarianceScaling(
                scale=beta, mode='fan_avg', distribution='truncated_normal', seed = 42)
            initializer = xavier_norm
        else:
            alpha = 1
            initializer = "glorot_normal"
        # multihead attention
        attn_output, attn_wts = self.mha(inputs,
                                         emb_size,
                                         d_model,
                                         num_heads,
                                         initializer,
                                         name=name + "/mha",
                                         layer_idx = layer_idx)
        # dropout for weight regularization
        attn_output = tf.keras.layers.Dropout(self.dropout_rate,
                                              name=name +
                                              "/dropout1")(attn_output)
        # skip connection + layernorm
        out1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=name + "/ln1")(inputs * alpha + attn_output)
        # feed-forward network
        ffn_output = self.ffn(out1,
                              emb_size,
                              dff,
                              initializer,
                              name=name + "/ffn")
        ffn_output = tf.keras.layers.Dropout(self.dropout_rate,
                                             name=name +
                                             "/dropout2")(ffn_output)
        out2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=name + "/ln2")(out1 * alpha + ffn_output)
        return out2, attn_wts

    def smolgen_weights(self,
                        inputs,
                        heads: int,
                        hidden_channels: int,
                        hidden_sz: int,
                        gen_sz: int,
                        name: str,
                        activation='swish'):
        compressed = tf.keras.layers.Dense(hidden_channels,
                                           name=name + '/compress',
                                           use_bias=False)(inputs)
        compressed = tf.reshape(compressed, [-1, 64 * hidden_channels])

        hidden = tf.keras.layers.Dense(hidden_sz,
                                       name=name + '/hidden1_dense',
                                       activation=activation)(compressed)
        hidden = tf.keras.layers.LayerNormalization(name=name +
                                                    '/hidden1_ln')(hidden)

        gen_from = tf.keras.layers.Dense(heads * gen_sz,
                                         name=name + '/gen_from',
                                         activation=activation)(hidden)
        gen_from = tf.keras.layers.LayerNormalization(name=name +
                                                      '/gen_from_ln')(gen_from)
        gen_from = tf.reshape(gen_from, [-1, heads, gen_sz])
        out = self.smol_weight_gen_dense(gen_from)
        return tf.reshape(out, [-1, heads, 64, 64])

    def create_residual_body(self, inputs):
        flow = self.conv_block(inputs,
                               filter_size=3,
                               output_channels=self.embedding_size,
                               name='input',
                               bn_scale=True)
        for i in range(self.RESIDUAL_BLOCKS):
            flow = self.residual_block(flow,
                                       self.embedding_size,
                                       name='residual_{}'.format(i + 1))
        return flow

    def create_encoder_body(self, inputs, embedding_size):
        # Policy head
        assert self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION

        # do some input processing
        if self.use_smolgen:
            self.smol_weight_gen_dense = tf.keras.layers.Dense(
                64 * 64, name='smol_weight_gen', use_bias=False)
            
        if self.embedding_style == "new":
            inputs = tf.cast(inputs, self.model_dtype)
            flow = tf.transpose(inputs, perm=[0, 2, 3, 1])
            flow = tf.reshape(flow, [-1, 64, tf.shape(inputs)[1]])

            # 1. Preprocess positional info
            pos_info = flow[..., :12]
            pos_info_flat = tf.reshape(pos_info, [-1, 64 * 12])

            pos_info_processed = tf.keras.layers.Dense(
                64 * self.embedding_dense_sz, name="embedding/preprocess")(pos_info_flat)
            pos_info = tf.reshape(pos_info_processed,
                                  [-1, 64, self.embedding_dense_sz])
            
            # Now both are float16, concat is safe
            flow = tf.keras.layers.Concatenate(axis=2)([flow, pos_info])

            # 2. Square embedding
            flow = tf.keras.layers.Dense(self.embedding_size, kernel_initializer="glorot_normal",
                                         activation=self.DEFAULT_ACTIVATION,
                                         name="embedding")(flow)
            
            # FIX: Use standard LayerNormalization instead of self.encoder_norm
            flow = tf.keras.layers.LayerNormalization(
                epsilon=1e-3, name="embedding/ln")(flow)
            flow = ma_gating(flow, name='embedding')

            # 3. DeepNorm scaling factors
            alpha = tf.cast(tf.math.pow(
                2. * self.encoder_layers, -0.25), self.model_dtype)
            beta = tf.cast(tf.math.pow(
                8. * self.encoder_layers, -0.25), self.model_dtype)

            xavier_norm = tf.keras.initializers.VarianceScaling(
                scale=beta, mode="fan_avg", distribution="truncated_normal", seed=42)

            # 4. Feed-forward network (FIX: removed 'activations =' unpacking)
            ffn_output = self.ffn(flow, self.embedding_size, self.encoder_dff,
                                  xavier_norm, name="embedding/ffn")

            # FIX: Use standard LayerNormalization instead of self.encoder_norm
            flow = tf.keras.layers.LayerNormalization(
                epsilon=1e-3, name="embedding/ffn_ln")(flow + ffn_output * alpha)

        else:
            flow = tf.transpose(inputs, perm=[0, 2, 3, 1])
            flow = tf.reshape(flow, [-1, 64, tf.shape(inputs)[1]])

            # add positional encoding for each square to the input
            if self.arc_encoding:
                self.POS_ENC = apm.make_pos_enc()
                positional_encoding = tf.broadcast_to(
                    tf.convert_to_tensor(self.POS_ENC, dtype=flow.dtype),
                    [tf.shape(flow)[0], 64,
                    tf.shape(self.POS_ENC)[2]])
                flow = tf.concat([flow, positional_encoding], axis=2)

            # square embedding
            flow = tf.keras.layers.Dense(embedding_size,
                                        kernel_initializer='glorot_normal',
                                        activation=self.DEFAULT_ACTIVATION,
                                        name='embedding')(flow)

            # !!! input gate
            flow = ma_gating(flow, name='embedding')
        attn_wts = []
        for i in range(self.encoder_layers):
            flow, attn_wts_l = self.encoder_layer(flow,
                                                  embedding_size,
                                                  self.encoder_d_model,
                                                  self.encoder_heads,
                                                  self.encoder_dff,
                                                  name='encoder_{}'.format(i +
                                                                           1),
                                                  layer_idx = i)
            attn_wts.append(attn_wts_l)
        return flow, attn_wts

    def apply_promotion_logits(self, queries, keys, attn_wts):
        # PAWN PROMOTION: create promotion logits using scalar offsets generated from the promotion-rank keys
        dk = tf.math.sqrt(tf.cast(tf.shape(keys)[-1],
                                  self.model_dtype))  # constant for scaling
        promotion_keys = keys[:, -8:, :]
        # queen, rook, bishop, knight order
        promotion_offsets = tf.keras.layers.Dense(
            4,
            kernel_initializer='glorot_normal',
            name='policy/attention/ppo',
            use_bias=False)(promotion_keys)
        promotion_offsets = tf.transpose(promotion_offsets,
                                         perm=[0, 2, 1]) * dk  # Bx4x8
        # knight offset is added to the other three
        promotion_offsets = promotion_offsets[:, :
                                              3, :] + promotion_offsets[:,
                                                                        3:4, :]

        # POLICY SELF-ATTENTION: self-attention weights are interpreted as from->to policy
        matmul_qk = tf.matmul(
            queries, keys,
            transpose_b=True)  # Bx64x64 (from 64 queries, 64 keys)

        # q, r, and b promotions are offset from the default promotion logit (knight)
        n_promo_logits = matmul_qk[:, -16:-8,
                                   -8:]  # default traversals from penultimate rank to promotion rank
        q_promo_logits = tf.expand_dims(n_promo_logits +
                                        promotion_offsets[:, 0:1, :],
                                        axis=3)  # Bx8x8x1
        r_promo_logits = tf.expand_dims(n_promo_logits +
                                        promotion_offsets[:, 1:2, :],
                                        axis=3)
        b_promo_logits = tf.expand_dims(n_promo_logits +
                                        promotion_offsets[:, 2:3, :],
                                        axis=3)
        promotion_logits = tf.concat(
            [q_promo_logits, r_promo_logits, b_promo_logits],
            axis=3)  # Bx8x8x3
        promotion_logits = tf.reshape(
            promotion_logits,
            [-1, 8, 24])  # logits now alternate a7a8q,a7a8r,a7a8b,...,

        # scale the logits by dividing them by sqrt(d_model) to stabilize gradients
        promotion_logits = promotion_logits / dk  # Bx8x24 (8 from-squares, 3x8 promotions)
        policy_attn_logits = matmul_qk / dk  # Bx64x64 (64 from-squares, 64 to-squares)

        attn_wts.append(promotion_logits)
        attn_wts.append(policy_attn_logits)

        # APPLY POLICY MAP: output becomes Bx1856
        h_fc1 = ApplyAttentionPolicyMap()(policy_attn_logits, promotion_logits)
        return h_fc1

    def construct_net(self, inputs, name=''):

        flow, attn_wts = self.create_encoder_body(inputs, self.embedding_size)


        assert (self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION)
        tokens = flow
        embed_activation = self.DEFAULT_ACTIVATION
        # SQUARE EMBEDDING: found to increase attention head performance
        tokens = tf.keras.layers.Dense(self.pol_embedding_size,
                                        kernel_initializer='glorot_normal',
                                        activation=embed_activation,
                                        name='policy/embedding')(tokens)

        # create queries and keys for policy self-attention
        queries = tf.keras.layers.Dense(self.policy_d_model,
                                        kernel_initializer='glorot_normal',
                                        name='policy/attention/wq')(tokens)
        keys = tf.keras.layers.Dense(self.policy_d_model,
                                        kernel_initializer='glorot_normal',
                                        name='policy/attention/wk')(tokens)

        h_fc1 = self.apply_promotion_logits(queries, keys, attn_wts)
        


        conv_val = tf.keras.layers.Dense(
            self.val_embedding_size,
            kernel_initializer='glorot_normal',
            activation=self.DEFAULT_ACTIVATION,
            name='value/embedding')(flow)


        h_conv_val_flat = tf.keras.layers.Flatten()(conv_val)
        h_fc2 = tf.keras.layers.Dense(128,
                                      kernel_initializer='glorot_normal',
                                      activation=self.DEFAULT_ACTIVATION,
                                      name='value/dense1')(h_conv_val_flat)
        if self.wdl:
            h_fc3 = tf.keras.layers.Dense(3,
                                          kernel_initializer='glorot_normal',
                                          dtype='float32',
                                          name='value/dense2')(h_fc2)
        else:
            h_fc3 = tf.keras.layers.Dense(1,
                                          kernel_initializer='glorot_normal',
                                          activation='tanh',
                                          dtype='float32',
                                          name='value/dense2')(h_fc2)

        # Moves left head
        if self.moves_left:
            conv_mov = tf.keras.layers.Dense(
                self.mov_embedding_size,
                kernel_initializer='glorot_normal',
                activation=self.DEFAULT_ACTIVATION,
                name='moves_left/embedding')(flow)

            h_conv_mov_flat = tf.keras.layers.Flatten()(conv_mov)
            h_fc4 = tf.keras.layers.Dense(
                128,
                kernel_initializer='glorot_normal',
                activation=self.DEFAULT_ACTIVATION,
                name='moves_left/dense1')(h_conv_mov_flat)

            h_fc5 = tf.keras.layers.Dense(1,
                                          kernel_initializer='glorot_normal',
                                          activation='relu',
                                          dtype='float32',
                                          name='moves_left/dense2')(h_fc4)
        else:
            h_fc5 = None

        if self.moves_left:
            outputs = [h_fc1, h_fc3, h_fc5, attn_wts]
        else:
            outputs = [h_fc1, h_fc3, attn_wts]

        return outputs
    '''