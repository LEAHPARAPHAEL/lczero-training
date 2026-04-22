def tf_name_to_pb_name(self, name):
        """Given Tensorflow variable name returns the protobuf name and index
        of residual block if weight belong in a residual block."""
        def value_to_bp(l, w):
            if l == 'dense_error':
                w = w.split(':')[0]
                d = {'kernel': 'ip_val_err_w', 'bias': 'ip_val_err_b'}
                return d[w]
            elif l == 'dense_cat':
                w = w.split(':')[0]
                d = {'kernel': 'ip_val_cat_w', 'bias': 'ip_val_cat_b'}
                return d[w]
            
            if l == 'embedding':
                n = ''
            elif l == 'dense1':
                n = 1
            elif l == 'dense2':
                n = 2
            else:
                raise ValueError('Unable to decode value weight {}/{}'.format(l, w))
            w = w.split(':')[0]
            d = {'kernel': 'ip{}_val_w', 'bias': 'ip{}_val_b'}
            return d[w].format(n)
        
        def attn_pol_to_bp(l, w):
            if l == 'wq':
                n = 2
            elif l == 'wk':
                n = 3
            elif l == 'ppo':
                n = 4
            else:
                raise ValueError('Unable to decode attn_policy weight {}/{}'.format(l, w))
            w = w.split(':')[0]
            d = {'kernel': 'ip{}_pol_w', 'bias': 'ip{}_pol_b'}
            return d[w].format(n)

        def encoder_to_bp(l, w):
            w = w.split(':')[0]
            d = {'gamma': '{}_gammas', 'beta': '{}_betas'}
            return d[w].format(l)

        def mha_to_bp(l, w):
            s = ''
            if l == 'quantize_1':
                return 's1'
            elif l == 'quantize_2':
                return 's2'
            elif l.startswith('rpe'):
                return l
            elif l.startswith('dense'):
                s = 'dense'
            elif l.startswith('w'):
                s = l[1]
            else:
                raise ValueError('Unable to decode mha weight {}/{}'.format(l, w))
            w = w.split(':')[0]
            d = {'kernel': '{}_w', 'bias': '{}_b', 's': '{}_s'}
            return d[w].format(s)

        def mha_smolgen_to_bp(l, w):
            s = {
                'compress': 'compress',
                'hidden1_dense': 'dense1_{}',
                'hidden1_ln': 'ln1_{}',
                'gen_from': 'dense2_{}',
                'gen_from_ln': 'ln2_{}'
            }
            if s.get(l) is None:
                raise ValueError('Unable to decode mha smolgen weight {}/{}'.format(l, w))
            w = w.split(':')[0]
            d = {
                'kernel': 'w',
                'bias': 'b',
                'gamma': 'gammas',
                'beta': 'betas'
            }
            return s[l].format(d[w])

        def ffn_to_bp(l, w):
            w = w.split(':')[0]
            if l == 'quantize_1':
                return 's1'
            elif l == 'quantize_2':
                return 's2'
            d = {'kernel': '{}_w', 'bias': '{}_b', 's': '{}_s'}
            return d[w].format(l)

        def moves_left_to_bp(l, w):
            if l == 'embedding':
                n = ''
            elif l == 'dense1':
                n = 1
            elif l == 'dense2':
                n = 2
            else:
                raise ValueError('Unable to decode moves_left weight {}/{}'.format(l, w))
            w = w.split(':')[0]
            d = {'kernel': 'ip{}_mov_w', 'bias': 'ip{}_mov_b'}
            return d[w].format(n)

        layers = name.split('/')
        base_layer = layers[0]
        weights_name = layers[-1]
        pb_name = None
        block = None
        encoder_block = None
        pol_encoder_block = None

        if base_layer == 'policy':
            pb_prefix = 'policy_heads.'
            if layers[1] == 'embedding':
                if layers[2].split(':')[0] == 'kernel':
                    pb_name = pb_prefix + 'ip_pol_w'
                else:
                    pb_name = pb_prefix + 'ip_pol_b'
            # STRIPPED: 'opponent' and 'next' are cleanly removed from this list
            elif layers[1] in ['vanilla', 'soft', 'optimistic_st']:
                pb_prefix = pb_prefix + layers[1] + '.'
                if layers[2] == 'attention':
                    pb_name = pb_prefix + attn_pol_to_bp(layers[3], weights_name)
            elif layers[1] == 'attention':
                pb_name = attn_pol_to_bp(layers[2], weights_name)

        elif base_layer == 'value':
            # FIX: Removed the unsafe index check for flat vs multihead values
            if layers[1] in ['st', 'q', 'winner']:
                pb_prefix = 'value_heads.' + layers[1] + '.'
                pb_name = pb_prefix + value_to_bp(layers[2], weights_name)
            else:
                pb_name = value_to_bp(layers[1], weights_name)

        elif base_layer == 'moves_left':
            if 'dense' in layers[1] or 'embedding' in layers[1]:
                pb_name = moves_left_to_bp(layers[1], weights_name)

        elif base_layer.startswith('encoder'):
            encoder_block = int(base_layer.split('_')[1]) - 1
            if layers[1] == 'mha':
                if layers[2] == 'smolgen':
                    pb_name = 'mha.smolgen.' + mha_smolgen_to_bp(layers[3], weights_name)
                else:
                    pb_name = 'mha.' + mha_to_bp(layers[2], weights_name)
            elif layers[1] == 'ffn':
                pb_name = 'ffn.' + ffn_to_bp(layers[2], weights_name)
            else:
                pb_name = encoder_to_bp(layers[1], weights_name)
                
        elif base_layer == 'embedding':
            if layers[1].split(':')[0] == 'kernel':
                pb_name = 'ip_emb_w'
            elif layers[1].split(':')[0] == 'bias':
                pb_name = 'ip_emb_b'
            elif layers[1] == 'ffn':
                pb_name = 'ip_emb_ffn.' + ffn_to_bp(layers[2], weights_name)
            elif layers[1] in ['ln', 'ffn_ln']:
                pb_name = 'ip_emb_' + encoder_to_bp(layers[1], weights_name)
            elif layers[1] == 'preprocess':
                if layers[2].split(':')[0] == 'kernel':
                    pb_name = 'ip_emb_preproc_w'
                else:
                    pb_name = 'ip_emb_preproc_b'
            if layers[1] == 'mult_gate' or layers[1] == 'add_gate':
                if layers[2].split(':')[0] == 'gate':
                    pb_name = 'ip_{}'.format(layers[1])

        elif base_layer == 'smol_weight_gen':
            if layers[1].split(':')[0] == 'kernel':
                pb_name = 'smolgen_w'
            else:
                pb_name = 'smolgen_b'
        
        else:
            raise ValueError('Unable to decode layer {}'.format(name))

        return (pb_name, block, pol_encoder_block, encoder_block)