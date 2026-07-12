#!/usr/bin/env python3
import unittest
import argparse
import sys
import os
import yaml
import numpy as np
import tensorflow as tf

# Import pipeline components
from train import get_latest_chunks, get_input_mode, game_number_for_name, identity_function, SKIP
from chunkparser import ChunkParser
from chunkparsefunc import parse_function
from tfprocess import TFProcess

class TestGatedBlockKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes the configuration, data parsers, and restores the trained model."""
        config_path = "./configs/GGT3.yaml"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}. Set the LEELA_CFG env var.")

        with open(config_path, "r") as f:
            cls.cfg = yaml.safe_load(f)

        # Dataset Setup (Mirrors train.py logic for test set sampling)
        num_chunks = cls.cfg["dataset"]["num_chunks"]
        allow_less = cls.cfg["dataset"].get("allow_less_chunks", False)
        train_ratio = cls.cfg["dataset"]["train_ratio"]
        fast_chunk_loading = cls.cfg["dataset"].get("fast_chunk_loading", True)
        num_train = int(num_chunks * train_ratio)
        num_test = num_chunks - num_train
        
        sort_type = cls.cfg["dataset"].get("sort_type", "mtime")
        sort_key_fn = {"mtime": os.path.getmtime, "number": game_number_for_name, "name": identity_function}[sort_type]

        if "input_test" in cls.cfg["dataset"]:
            test_chunks = get_latest_chunks(cls.cfg["dataset"]["input_test"], num_test, allow_less, sort_key_fn, fast=fast_chunk_loading)
        else:
            chunks = get_latest_chunks(cls.cfg["dataset"]["input"], num_chunks, allow_less, sort_key_fn, fast=fast_chunk_loading)
            test_chunks = chunks[int(len(chunks) * train_ratio):]

        # Parser configuration
        total_batch_size = cls.cfg["training"]["batch_size"]
        batch_splits = cls.cfg["training"].get("num_batch_splits", 1)
        split_batch_size = total_batch_size // batch_splits
        test_shuffle_size = int(cls.cfg["training"]["shuffle_size"] * (1.0 - train_ratio))

        cls.test_parser = ChunkParser(
            test_chunks,
            get_input_mode(cls.cfg),
            shuffle_size=test_shuffle_size,
            sample=SKIP,
            batch_size=split_batch_size,
            workers=cls.cfg["dataset"].get("test_workers", 0)
        )

        cls.tfprocess = TFProcess(cls.cfg)
        output_types = 8 * (tf.string,)
        test_dataset = tf.data.Dataset.from_generator(cls.test_parser.parse, output_types=output_types).map(parse_function)
        
        cls.tfprocess.init(test_dataset, test_dataset, None)
        # Suppress downstream validation warnings from partial check restorations
        cls.tfprocess.restore()

        cls.groups = cls.cfg['model'].get('gating_groups', 1)
        cls.test_iter = iter(test_dataset)

    @classmethod
    def tearDownClass(cls):
        """Safely shuts down the parser workers, bypassing internal mutation index errors."""
        try:
            cls.test_parser.shutdown()
        except IndexError:
            # Handles the chunkparser.py bug where readers are dropped on EOF
            pass

    def test_extract_and_display_recalibrated_kernels(self):
        """Extracts chess positions from input planes and computes actual modulated kernels."""
        print("\n" + "="*80)
        print("  GATED BLOCK DEPTHWISE KERNEL PHYSICAL RECALIBRATION PROFILE")
        print("="*80)

        # 1. Grab a production evaluation batch
        x, _, _, _, _, _, _, _ = next(self.test_iter)

        # 2. Locate the output layers for gating masks
        gating_out_layers = [l for l in self.tfprocess.model.layers if "gating/out" in l.name]
        
        if not gating_out_layers:
            self.skipTest("No Gated Blocks ('G') detected in the active model architecture configurations.")

        # 3. Create an inspection sub-model to get raw gating outputs
        inspection_model = tf.keras.Model(
            inputs=self.tfprocess.model.input,
            outputs={layer.name: layer.output for layer in gating_out_layers}
        )
        raw_gating_activations = inspection_model(x, training=False)

        max_positions_to_show = 1
        max_groups_to_show = min(4, self.groups)
        
        # Piece configuration for the first 12 binary configuration planes
        piece_symbols = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]

        for layer_name, raw_gating in raw_gating_activations.items():
            block_prefix = layer_name.split("/gating/out")[0]
            dw_layer_name = f"{block_prefix}/2/conv2d"
            
            try:
                dw_layer = self.tfprocess.model.get_layer(dw_layer_name)
            except ValueError:
                continue

            dw_kernel_param = [w for w in dw_layer.weights if "depthwise_kernel" in w.name]
            if not dw_kernel_param:
                continue
            
            # FIXED: Handle flattened (25, C) physical shape directly
            base_kernel_np = dw_kernel_param[0].numpy() 
            dff = base_kernel_np.shape[1]  # Dimension index 1 holds channels (C)
            channels_per_group = dff // self.groups
            base_kernel_flat = base_kernel_np  # No reshape needed, shape is already (25, C)

            # Apply sigmoid activation to match backend custom layer inputs
            gating_weights = tf.sigmoid(raw_gating).numpy()
            if len(gating_weights.shape) == 2:
                gating_weights = np.reshape(gating_weights, [-1, 25, self.groups])

            print(f"\n[Gated Block]: {block_prefix} (Channels: {dff}, Groups: {self.groups})")
            print(f"  ↳ Base Depthwise Layer: {dw_layer_name}")

            for pos_idx in range(min(max_positions_to_show, tf.shape(x)[0].numpy())):
                print(f"\n    ♟ Position Sample Index #{pos_idx + 1}")
                print("    " + "-"*50)
                
                # --- RECONSTRUCT CHESS BOARD FROM PLANES 0-11 ---
                # x layout is [Batch, Planes, Height, Width]
                input_planes = x[pos_idx, :12, :, :].numpy() 
                
                print("    CURRENT POSITION STATE (Planes 0-11)")
                print("      +-----------------+")
                for r in range(8):  # Rows 8 down to 1
                    row_str = []
                    for c in range(8):  # Files a through h
                        active_symbol = "."
                        for plane_idx in range(12):
                            if input_planes[plane_idx, r, c] == 1.0:
                                active_symbol = piece_symbols[plane_idx]
                                break
                        row_str.append(active_symbol)
                    print(f"    {8 - r} | {' '.join(row_str)} |")
                print("      +-----------------+")
                print("        a b c d e f g h\n")
                
                # --- PROCESS AND MULTIPLY KERNELS ---
                for group_idx in range(max_groups_to_show):
                    group_gating_vector = gating_weights[pos_idx, :, group_idx]
                    rep_channel = group_idx * channels_per_group
                    
                    # Slicing from the (25, C) matrix directly
                    static_channel_kernel = base_kernel_flat[:, rep_channel]
                    
                    # Element-wise product matching CUDA engine operations
                    recalibrated_kernel = static_channel_kernel * group_gating_vector
                    matrix_5x5 = np.reshape(recalibrated_kernel, [5, 5])
                    
                    print(f"      • Group {group_idx} (Channel Reference #{rep_channel}) Physical 5x5 Kernel:")
                    for row in matrix_5x5:
                        row_str = " ".join([f"{val:7.4f}" for val in row])
                        print(f"          [ {row_str} ]")
                    print()
        
        print("="*80 + "\n")

if __name__ == "__main__":
    unittest.main()