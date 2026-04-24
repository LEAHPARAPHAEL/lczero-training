#!/usr/bin/env python3
#
#    This file is part of Leela Chess.
#    Copyright (C) 2021 Leela Chess Authors
#
#    Leela Chess is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Chess is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
import tensorflow as tf

'''
def parse_function(planes, probs, winner, q, plies_left, st_q, opp_probs, next_probs, fut):
    """
    Convert unpacked record batches to tensors for tensorflow training
    """
    planes = tf.io.decode_raw(planes, tf.float32)
    probs = tf.io.decode_raw(probs, tf.float32)
    winner = tf.io.decode_raw(winner, tf.float32)
    q = tf.io.decode_raw(q, tf.float32)
    plies_left = tf.io.decode_raw(plies_left, tf.float32)
    st_q = tf.io.decode_raw(st_q, tf.float32)
    opp_probs = tf.io.decode_raw(opp_probs, tf.float32)
    next_probs = tf.io.decode_raw(next_probs, tf.float32)
    fut = tf.io.decode_raw(fut, tf.float32)


    planes = tf.reshape(planes, (-1, 112, 8, 8))
    probs = tf.reshape(probs, (-1, 1858))
    winner = tf.reshape(winner, (-1, 3))
    q = tf.reshape(q, (-1, 3))
    plies_left = tf.reshape(plies_left, (-1, 1))
    st_q = tf.reshape(st_q, (-1, 3))
    opp_probs = tf.reshape(opp_probs, (-1, 1858))
    next_probs = tf.reshape(next_probs, (-1, 1858))


    fut = tf.reshape(fut, (-1, 16, 12, 64))
    fut = tf.transpose(fut, perm=[0, 3, 1, 2])
    fut = tf.concat([fut, 1 - tf.reduce_sum(fut, axis=-1, keepdims=True)], axis=-1)


    return (planes, probs, winner, q, plies_left, st_q, opp_probs, next_probs, fut)
'''

def parse_function(planes, probs, winner, root_wdl, plies_left, st_wdl, opp_probs, next_probs):
    """
    Convert unpacked record batches to tensors for tensorflow training
    """
    # 1. Decode standard float targets
    planes = tf.io.decode_raw(planes, tf.float32)
    probs = tf.io.decode_raw(probs, tf.float32)
    winner = tf.io.decode_raw(winner, tf.float32)
    root_wdl = tf.io.decode_raw(root_wdl, tf.float32)
    plies_left = tf.io.decode_raw(plies_left, tf.float32)
    st_wdl = tf.io.decode_raw(st_wdl, tf.float32)

    opp_probs = tf.io.decode_raw(opp_probs, tf.float32)
    next_probs = tf.io.decode_raw(next_probs, tf.float32)
    # 3. Reshape floats
    planes = tf.reshape(planes, (-1, 112, 8, 8))
    probs = tf.reshape(probs, (-1, 1858))
    winner = tf.reshape(winner, (-1, 3))
    root_wdl = tf.reshape(root_wdl, (-1, 3))
    plies_left = tf.reshape(plies_left, (-1, 1))
    st_wdl = tf.reshape(st_wdl, (-1, 3))
    
    opp_probs = tf.reshape(opp_probs, (-1, 1858))
    next_probs = tf.reshape(next_probs, (-1, 1858))

    # Return exactly the 8 variables that process_inner_loop expects!
    # x=planes, y=probs, z=winner, q=root_wdl, m=plies_left, st_q=st_wdl, opp_idx=opp_idx, next_idx=next_idx
    return planes, probs, winner, root_wdl, plies_left, st_wdl, opp_probs, next_probs