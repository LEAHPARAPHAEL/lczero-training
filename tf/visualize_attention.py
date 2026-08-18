#!/usr/bin/env python3
import os
import yaml
import numpy as np
import tensorflow as tf
import pygame
import sys
import random
import argparse

# Import components from training pipeline
import chunkparser
from train import get_latest_chunks, get_input_mode, game_number_for_name, identity_function, SKIP
from chunkparser import ChunkParser
from chunkparsefunc import parse_function
from tfprocess import TFProcess


# --- CONFIGURATION & PIPELINE INITIALIZATION ---
def init_pipeline(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    num_chunks = cfg["dataset"]["num_chunks"]
    allow_less = cfg["dataset"].get("allow_less_chunks", False)
    train_ratio = cfg["dataset"]["train_ratio"]
    fast_chunk_loading = cfg["dataset"].get("fast_chunk_loading", True)
    num_train = int(num_chunks * train_ratio)
    num_test = num_chunks - num_train
    
    sort_type = cfg["dataset"].get("sort_type", "mtime")
    sort_key_fn = {"mtime": os.path.getmtime, "number": game_number_for_name, "name": identity_function}[sort_type] if sort_type != "mtime" else os.path.getmtime

    if "input_test" in cfg["dataset"]:
        test_chunks = get_latest_chunks(cfg["dataset"]["input_test"], num_test, allow_less, sort_key_fn, fast=fast_chunk_loading)
    else:
        chunks = get_latest_chunks(cfg["dataset"]["input"], num_chunks, allow_less, sort_key_fn, fast=fast_chunk_loading)
        test_chunks = chunks[int(len(chunks) * train_ratio):]

    # Deterministic chunk ordering
    test_chunks = sorted(test_chunks)

    total_batch_size = cfg["training"]["batch_size"]
    batch_splits = cfg["training"].get("num_batch_splits", 1)
    split_batch_size = total_batch_size // batch_splits

    # Single-threaded, deterministic chunk parser without shuffle buffer
    parser = ChunkParser(
        test_chunks, 
        get_input_mode(cfg), 
        shuffle_size=0, 
        sample=1, 
        batch_size=split_batch_size, 
        workers=0
    )

    tfprocess = TFProcess(cfg)
    
    # Standard 8-string dataset generation
    test_dataset = tf.data.Dataset.from_generator(
        parser.parse, output_types=8 * (tf.string,)
    ).map(parse_function)
    
    tfprocess.init(test_dataset, test_dataset, None)
    tfprocess.restore()

    return iter(test_dataset), tfprocess, parser


# --- DATA EXTRACTION & ATTENTION PROBE ENGINE ---
class AttentionPositionState:
    def __init__(self, test_iter, tfprocess):
        self.test_iter = test_iter
        self.tfprocess = tfprocess
        self.batch_x = None
        self.batch_size = 0
        self.current_idx = -1
        
        self.history = []
        self.history_pointer = -1

        self.filters = {
            'P': {'min': 0, 'max': 16},
            'N': {'min': 0, 'max': 16},
            'B': {'min': 0, 'max': 16},
            'R': {'min': 0, 'max': 16},
            'Q': {'min': 0, 'max': 16}
        }
        
        self.layer_info = []
        enc_idx = 0
        for block_idx, block_type in enumerate(self.tfprocess.blocks):
            if block_type in ['T', 'B', 'D']:
                name = f"block_{block_idx}_encoder (Layer {enc_idx})"
                self.layer_info.append({
                    'name': name,
                    'block_idx': block_idx,
                    'enc_idx': enc_idx,
                    'prefix': f"block_{block_idx}_encoder/mha"
                })
                enc_idx += 1
                
        if not self.layer_info:
            print("Error: Active architecture does not contain Transformer/Attention blocks ('T', 'B', or 'D').")
            sys.exit(1)
            
        self.layer_names = [info['name'] for info in self.layer_info]
        self.active_layer_idx = 0
        self.selected_square = (4, 4) # Default selected token: e4/d4

        # Build non-intrusive probe model to extract Q & K tensors
        probe_outputs = {}
        for info in self.layer_info:
            p = info['prefix']
            
            rope_layer = None
            for l in self.tfprocess.model.layers:
                if l.name == f"{p}/rope":
                    rope_layer = l
                    break
            
            if rope_layer is not None:
                probe_outputs[f"{p}/q"] = rope_layer.output[0]
                probe_outputs[f"{p}/k"] = rope_layer.output[1]
            else:
                wq_layer = self.tfprocess.model.get_layer(f"{p}/wq")
                wk_layer = self.tfprocess.model.get_layer(f"{p}/wk")
                probe_outputs[f"{p}/q_raw"] = wq_layer.output
                probe_outputs[f"{p}/k_raw"] = wk_layer.output

            if self.tfprocess.use_rpe_q:
                rpe_q_layer = self.tfprocess.model.get_layer(f"{p}/rpe_q")
                probe_outputs[f"{p}/rpe_q"] = rpe_q_layer.output
            if self.tfprocess.use_rpe_k:
                rpe_k_layer = self.tfprocess.model.get_layer(f"{p}/rpe_k")
                probe_outputs[f"{p}/rpe_k"] = rpe_k_layer.output

            if self.tfprocess.use_smolgen:
                smol_gen_ln = self.tfprocess.model.get_layer(f"{p}/smolgen/gen_from_ln")
                probe_outputs[f"{p}/smolgen_gen_from_ln"] = smol_gen_ln.output

        self.probe_model = tf.keras.Model(
            inputs=self.tfprocess.model.input,
            outputs=probe_outputs
        )

    def matches_filter(self, board):
        counts = {'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0}
        for row in board:
            for piece in row:
                u_piece = piece.upper()
                if u_piece in counts:
                    counts[u_piece] += 1
        
        for p_type, limits in self.filters.items():
            if not (limits['min'] <= counts[p_type] <= limits['max']):
                return False
        return True

    def advance_position(self):
        if self.history_pointer < len(self.history) - 1:
            self.history_pointer += 1
            return self.history[self.history_pointer]

        attempts = 0
        while attempts < 5000:
            if self.batch_x is None or self.current_idx >= self.batch_size - 1:
                print("Fetching next dataset batch array from stream...")
                try:
                    outputs = next(self.test_iter)
                except StopIteration:
                    print("End of dataset reached.")
                    return None
                
                self.batch_x = outputs[0]
                
                # Forward pass through probe model
                probed = self.probe_model(self.batch_x, training=False)
                b_size = tf.shape(self.batch_x)[0]
                num_heads = self.tfprocess.encoder_heads
                d_model = self.tfprocess.encoder_d_model
                head_depth = d_model // num_heads

                self.batch_attn_dict = {}
                for info in self.layer_info:
                    p = info['prefix']
                    enc_idx = info['enc_idx']
                    
                    if f"{p}/q" in probed:
                        q = probed[f"{p}/q"]
                        k = probed[f"{p}/k"]
                    else:
                        q_raw = probed[f"{p}/q_raw"]
                        k_raw = probed[f"{p}/k_raw"]
                        q = self.tfprocess.split_heads(q_raw, b_size, num_heads, head_depth)
                        k = self.tfprocess.split_heads(k_raw, b_size, num_heads, head_depth)

                    dk = tf.cast(tf.shape(k)[-1], q.dtype)
                    matmul_qk = tf.matmul(q, k, transpose_b=True)

                    if self.tfprocess.use_rpe_q:
                        matmul_qk = matmul_qk + probed[f"{p}/rpe_q"]
                    if self.tfprocess.use_rpe_k:
                        matmul_qk = matmul_qk + probed[f"{p}/rpe_k"]

                    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

                    if hasattr(self.tfprocess, 'static_attn_bias') and self.tfprocess.use_static_bias:
                        layer_bias = self.tfprocess.static_attn_bias[enc_idx, :, :num_heads, :, :]
                        scaled_attention_logits = scaled_attention_logits + tf.cast(layer_bias, scaled_attention_logits.dtype)

                    if self.tfprocess.use_smolgen:
                        gen_from = probed[f"{p}/smolgen_gen_from_ln"]
                        gen_from = tf.reshape(gen_from, [-1, num_heads, self.tfprocess.smolgen_gen_sz])
                        smol_out = self.tfprocess.smol_weight_gen_dense(gen_from)
                        smol_weights = tf.reshape(smol_out, [-1, num_heads, 64, 64])
                        scaled_attention_logits = scaled_attention_logits + tf.cast(smol_weights, scaled_attention_logits.dtype)

                    if self.tfprocess.use_logit_gating:
                        rel_bias_layer = self.tfprocess.model.get_layer(f"{p}/rel_bias")
                        scaled_attention_logits = rel_bias_layer(scaled_attention_logits)

                    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
                    self.batch_attn_dict[info['name']] = attention_weights.numpy()

                self.batch_size = self.batch_x.shape[0]
                self.current_idx = 0
            else:
                self.current_idx += 1
            
            attempts += 1
            board, attn_dict, is_black_to_move = self.get_current_data()
            if self.matches_filter(board):
                self.history.append((board, attn_dict, is_black_to_move))
                self.history_pointer += 1
                return board, attn_dict, is_black_to_move
        
        print("Warning: 5000 positions scanned. No match found for current filters.")
        return None

    def previous_position(self):
        if self.history_pointer > 0:
            self.history_pointer -= 1
        return self.history[self.history_pointer]

    def get_current_data(self):
        # Plane 108 indicates if Black is to move (1.0 for Black, 0.0 for White)
        is_black_to_move = bool(self.batch_x[self.current_idx, 108, 0, 0].numpy() > 0.5)

        if not is_black_to_move:
            piece_symbols = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
        else:
            piece_symbols = ["p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]

        board_matrix = [["." for _ in range(8)] for _ in range(8)]
        input_planes = self.batch_x[self.current_idx, :12, :, :].numpy()

        for r in range(8):
            for c in range(8):
                for p_idx in range(12):
                    if input_planes[p_idx, r, c] == 1.0:
                        board_matrix[r][c] = piece_symbols[p_idx]
                        break

        attn_dict = {}
        for name in self.layer_names:
            if name in self.batch_attn_dict:
                attn_dict[name] = self.batch_attn_dict[name][self.current_idx]

        return board_matrix, attn_dict, is_black_to_move


# --- VIRIDIS COLORMAP GENERATOR ---
def viridis_colormap(val):
    v = np.clip(val, 0.0, 1.0)
    colors = [
        (0.00, (68, 1, 84)),     # Deep Purple
        (0.25, (59, 82, 139)),   # Dark Blue
        (0.50, (33, 145, 140)),   # Teal
        (0.75, (94, 201, 98)),   # Emerald Green
        (1.00, (253, 231, 37))   # Radiant Yellow
    ]
    for i in range(len(colors) - 1):
        v0, c0 = colors[i]
        v1, c1 = colors[i+1]
        if v <= v1:
            t = (v - v0) / (v1 - v0)
            r = int(c0[0] + t * (c1[0] - c0[0]))
            g = int(c0[1] + t * (c1[1] - c0[1]))
            b = int(c0[2] + t * (c1[2] - c0[2]))
            return (r, g, b)
    return (253, 231, 37)


# --- INTERFACE RENDERER ---
def draw_interface(screen, fonts, board, attn_dict, active_layer_name, selected_square, is_black_to_move, 
                   prev_btn_rect, next_btn_rect, filter_btn_rect, 
                   prev_layer_rect, next_layer_rect, scroll_y):
    # Set crisp white background
    screen.fill((255, 255, 255))

    if not is_black_to_move:
        rank_labels = [str(8 - i) for i in range(8)]
        file_labels = [chr(97 + i) for i in range(8)]
    else:
        rank_labels = [str(i + 1) for i in range(8)]
        file_labels = [chr(104 - i) for i in range(8)]

    mouse_pos = pygame.mouse.get_pos()
    
    # 1. Top Controls Bar
    for btn, color_base, text in [(prev_layer_rect, (55, 60, 70), "◀ PREV LAYER"), 
                                  (next_layer_rect, (55, 60, 70), "NEXT LAYER ▶")]:
        color = tuple(min(c + 25, 255) for c in color_base) if btn.collidepoint(mouse_pos) else color_base
        pygame.draw.rect(screen, color, btn, border_radius=4)
        lbl = fonts['sub_bold'].render(text, True, (255, 255, 255))
        screen.blit(lbl, lbl.get_rect(center=btn.center))

    layer_title = fonts['main'].render(f"{active_layer_name}", True, (20, 25, 35))
    screen.blit(layer_title, layer_title.get_rect(center=(760, 36)))

    sel_r, sel_c = selected_square
    sel_sq_name = f"{file_labels[sel_c]}{rank_labels[sel_r]}".upper()
    sel_piece = board[sel_r][sel_c]
    piece_names = {'P': 'White Pawn', 'N': 'White Knight', 'B': 'White Bishop', 'R': 'White Rook', 'Q': 'White Queen', 'K': 'White King',
                   'p': 'Black Pawn', 'n': 'Black Knight', 'b': 'Black Bishop', 'r': 'Black Rook', 'q': 'Black Queen', 'k': 'Black King', '.': 'Empty Square'}
    
    token_lbl = fonts['sub_bold'].render(
        f"Active Query Token: {sel_sq_name} [{piece_names.get(sel_piece, 'Empty')}]  (Click any square on any board to re-focus)", 
        True, (160, 95, 0)
    )
    screen.blit(token_lbl, token_lbl.get_rect(center=(760, 68)))

    # 2. Scrollable 4-Column Grid Viewport
    viewport_rect = pygame.Rect(30, 95, 1445, 755)
    screen.set_clip(viewport_rect)

    layer_attn = attn_dict.get(active_layer_name, None)
    sq_size = 38
    board_px = sq_size * 8
    spacing_x = 356
    spacing_y = 330
    start_x = 45
    start_y = 105

    unicode_pieces = {'K': '♚', 'Q': '♛', 'R': '♜', 'B': '♝', 'N': '♞', 'P': '♟',
                      'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'}

    if layer_attn is not None:
        num_heads = layer_attn.shape[0]
        token_idx = sel_r * 8 + sel_c

        for h in range(num_heads):
            gx = h % 4
            gy = h // 4
            bx = start_x + gx * spacing_x
            by = start_y + gy * spacing_y + scroll_y

            if by + spacing_y < viewport_rect.top or by > viewport_rect.bottom:
                continue

            head_weights = layer_attn[h, token_idx, :].reshape(8, 8)
            max_val = float(np.max(head_weights))

            # Board Grid
            for r in range(8):
                for c in range(8):
                    rect = pygame.Rect(bx + c * sq_size, by + r * sq_size, sq_size, sq_size)
                    raw_w = head_weights[r, c]
                    norm_w = raw_w / (max_val + 1e-7) if max_val > 0 else 0.0
                    cell_color = viridis_colormap(norm_w)

                    pygame.draw.rect(screen, cell_color, rect)
                    pygame.draw.rect(screen, (140, 130, 175), rect, 1)

                    # Query Square Crimson Box
                    if (r, c) == selected_square:
                        pygame.draw.rect(screen, (220, 35, 45), rect, 3)

                    # Piece Glyph with Contrasting Outline
                    piece = board[r][c]
                    if piece != ".":
                        glyph = unicode_pieces.get(piece, piece)
                        cx = bx + c * sq_size + sq_size // 2
                        cy = by + r * sq_size + sq_size // 2

                        if piece.isupper():
                            for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                                shadow_text = fonts['piece'].render(glyph, True, (0, 0, 0))
                                screen.blit(shadow_text, shadow_text.get_rect(center=(cx + ox, cy + oy)))
                            main_text = fonts['piece'].render(glyph, True, (255, 255, 255))
                            screen.blit(main_text, main_text.get_rect(center=(cx, cy)))
                        else:
                            for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                                border_text = fonts['piece'].render(glyph, True, (255, 255, 255))
                                screen.blit(border_text, border_text.get_rect(center=(cx + ox, cy + oy)))
                            main_text = fonts['piece'].render(glyph, True, (10, 10, 10))
                            screen.blit(main_text, main_text.get_rect(center=(cx, cy)))

            # Outer Border
            pygame.draw.rect(screen, (40, 45, 55), (bx, by, board_px, board_px), 2)

    screen.set_clip(None)

    # 3. Dynamic Scrollbar Indicator
    if layer_attn is not None:
        total_rows = (layer_attn.shape[0] + 3) // 4
        content_height = total_rows * spacing_y + 40
        viewport_h = viewport_rect.height
        if content_height > viewport_h:
            track_rect = pygame.Rect(1485, 95, 10, viewport_h)
            pygame.draw.rect(screen, (230, 232, 238), track_rect, border_radius=5)
            
            thumb_ratio = viewport_h / content_height
            thumb_h = max(30, int(viewport_h * thumb_ratio))
            scroll_ratio = -scroll_y / (content_height - viewport_h)
            thumb_y = 95 + int(scroll_ratio * (viewport_h - thumb_h))
            
            thumb_rect = pygame.Rect(1485, thumb_y, 10, thumb_h)
            pygame.draw.rect(screen, (160, 165, 175), thumb_rect, border_radius=5)

    # 4. Bottom Controls Bar
    for btn, color_base, text in [(prev_btn_rect, (85, 90, 100), "PREV POS"), 
                                  (next_btn_rect, (45, 125, 75), "NEXT POS"), 
                                  (filter_btn_rect, (40, 95, 140), "FILTERS")]:
        color = tuple(min(c + 25, 255) for c in color_base) if btn.collidepoint(mouse_pos) else color_base
        pygame.draw.rect(screen, color, btn, border_radius=6)
        lbl = fonts['sub_bold'].render(text, True, (255, 255, 255))
        screen.blit(lbl, lbl.get_rect(center=btn.center))


def draw_filter_dialog(screen, fonts, filters, reset_btn_rect, close_btn_rect, active_handle):
    overlay = pygame.Surface((1520, 920), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))
    screen.blit(overlay, (0, 0))

    panel_rect = pygame.Rect(510, 220, 500, 440)
    pygame.draw.rect(screen, (245, 247, 250), panel_rect, border_radius=12)
    pygame.draw.rect(screen, (180, 185, 195), panel_rect, width=2, border_radius=12)

    title = fonts['main'].render("OCCURRENCE FILTERS CONFIGURATION", True, (25, 30, 40))
    screen.blit(title, title.get_rect(center=(760, 260)))

    piece_names = {'P': 'Pawns (P)', 'N': 'Knights (N)', 'B': 'Bishops (B)', 'R': 'Rooks (R)', 'Q': 'Queens (Q)'}
    
    track_x, track_w = 680, 240
    for idx, (p_code, name) in enumerate(piece_names.items()):
        row_y = 320 + idx * 52
        
        lbl = fonts['sub_bold'].render(name, True, (40, 45, 55))
        screen.blit(lbl, (540, row_y - 8))

        pygame.draw.line(screen, (190, 195, 205), (track_x, row_y), (track_x + track_w, row_y), 6)
        
        min_val, max_val = filters[p_code]['min'], filters[p_code]['max']
        min_x = track_x + int((min_val / 16) * track_w)
        max_x = track_x + int((max_val / 16) * track_w)

        pygame.draw.line(screen, (50, 140, 85), (min_x, row_y), (max_x, row_y), 6)

        pygame.draw.circle(screen, (220, 80, 80), (min_x, row_y), 8)
        pygame.draw.circle(screen, (80, 180, 120), (max_x, row_y), 8)

        val_lbl = fonts['label'].render(f"Min: {min_val}  Max: {max_val}", True, (70, 75, 85))
        screen.blit(val_lbl, (680, row_y + 10))

    mouse_pos = pygame.mouse.get_pos()
    
    reset_color = (165, 75, 75) if reset_btn_rect.collidepoint(mouse_pos) else (140, 60, 60)
    pygame.draw.rect(screen, reset_color, reset_btn_rect, border_radius=6)
    r_lbl = fonts['sub_bold'].render("RESET", True, (255, 255, 255))
    screen.blit(r_lbl, r_lbl.get_rect(center=reset_btn_rect.center))
    
    btn_color = (65, 145, 88) if close_btn_rect.collidepoint(mouse_pos) else (48, 118, 70)
    pygame.draw.rect(screen, btn_color, close_btn_rect, border_radius=6)
    c_lbl = fonts['sub_bold'].render("APPLY & CLOSE", True, (255, 255, 255))
    screen.blit(c_lbl, c_lbl.get_rect(center=close_btn_rect.center))


# --- MAIN RUNNER LOOP ---
def main():
    parser = argparse.ArgumentParser(description="Leela Chess Multi-Head Attention Inspector")
    parser.add_argument("--cfg", "--config", "-c", dest="config", default="configs/BT4.yaml",
                        help="Path to the model YAML configuration file")
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.exists(config_path):
        if os.path.exists(f"./configs/{args.config}"):
            config_path = f"./configs/{args.config}"
        elif os.path.exists(f"./configs/{args.config}.yaml"):
            config_path = f"./configs/{args.config}.yaml"
        else:
            raise FileNotFoundError(f"Could not find config file: {args.config}")

    print(f"Loading configuration from: {config_path}")
    
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    os.environ["SDL_AUDIODRIVER"] = "dummy"
    pygame.display.init()
    pygame.font.init()

    screen = pygame.display.set_mode((1520, 920))
    pygame.display.set_caption("Leela Chess Multi-Head Attention Inspector")
    
    fallback_fonts = ['segoeuisymbol', 'dejavusans', 'freesans', 'arialunicodems', 'sans-serif']
    fonts = {
        'main': pygame.font.SysFont('sans-serif', 20, bold=True),
        'sub_bold': pygame.font.SysFont('sans-serif', 13, bold=True),
        'label': pygame.font.SysFont('monospace', 11, bold=True),
        'piece': pygame.font.SysFont(fallback_fonts, 28),
    }

    print("Booting reproducible pipeline streaming engines...")
    test_iter, tfprocess, chunk_parser = init_pipeline(config_path)
    
    state = AttentionPositionState(test_iter, tfprocess)
    res = state.advance_position()
    board, attn_dict, is_black_to_move = res if res else ([["."]*8]*8, {}, False)

    # UI Geometry
    prev_layer_rect = pygame.Rect(430, 20, 130, 32)
    next_layer_rect = pygame.Rect(960, 20, 130, 32)
    
    prev_btn_rect = pygame.Rect(540, 865, 130, 42)
    next_btn_rect = pygame.Rect(695, 865, 130, 42)
    filter_btn_rect = pygame.Rect(850, 865, 130, 42)

    dialog_reset_rect = pygame.Rect(540, 595, 190, 40)
    dialog_close_rect = pygame.Rect(790, 595, 190, 40)

    sq_size = 38
    spacing_x = 356
    spacing_y = 330
    start_x = 45
    start_y = 105
    viewport_rect = pygame.Rect(30, 95, 1445, 755)

    scroll_y = 0
    show_filters = False
    active_handle = None 
    clock = pygame.time.Clock()

    while True:
        active_layer_name = state.layer_names[state.active_layer_idx]
        layer_attn = attn_dict.get(active_layer_name, None)
        num_heads = layer_attn.shape[0] if layer_attn is not None else 8
        total_rows = (num_heads + 3) // 4
        content_height = total_rows * spacing_y + 40
        min_scroll_y = min(0, viewport_rect.height - content_height)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                try: chunk_parser.shutdown()
                except (IndexError, AttributeError): pass
                pygame.quit(); sys.exit()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if show_filters:
                    if dialog_close_rect.collidepoint(event.pos):
                        show_filters = False
                        active_handle = None
                    elif dialog_reset_rect.collidepoint(event.pos):
                        for p_code in state.filters:
                            state.filters[p_code]['min'] = 0
                            state.filters[p_code]['max'] = 16
                    else:
                        track_x, track_w = 680, 240
                        for idx, p_code in enumerate(['P', 'N', 'B', 'R', 'Q']):
                            row_y = 320 + idx * 52
                            min_val = state.filters[p_code]['min']
                            max_val = state.filters[p_code]['max']
                            min_x = track_x + int((min_val / 16) * track_w)
                            max_x = track_x + int((max_val / 16) * track_w)
                            
                            dist_min = np.hypot(event.pos[0] - min_x, event.pos[1] - row_y)
                            dist_max = np.hypot(event.pos[0] - max_x, event.pos[1] - row_y)
                            
                            if dist_min < 12 or dist_max < 12:
                                if min_val == max_val:
                                    if min_val == 0: active_handle = (p_code, 'max')
                                    elif min_val == 16: active_handle = (p_code, 'min')
                                    else: active_handle = (p_code, 'max')
                                else:
                                    if dist_min < dist_max: active_handle = (p_code, 'min')
                                    else: active_handle = (p_code, 'max')
                                break
                else:
                    if event.button == 1:
                        # Global Selection across visible boards
                        if viewport_rect.collidepoint(event.pos):
                            clicked_any = False
                            for h in range(num_heads):
                                gx = h % 4
                                gy = h // 4
                                bx = start_x + gx * spacing_x
                                by = start_y + gy * spacing_y + scroll_y
                                board_rect = pygame.Rect(bx, by, sq_size * 8, sq_size * 8)
                                
                                if board_rect.collidepoint(event.pos):
                                    click_c = (event.pos[0] - bx) // sq_size
                                    click_r = (event.pos[1] - by) // sq_size
                                    if 0 <= click_r < 8 and 0 <= click_c < 8:
                                        state.selected_square = (click_r, click_c)
                                        clicked_any = True
                                        break
                            if clicked_any:
                                continue

                        # Layer & Navigation Controls
                        if prev_layer_rect.collidepoint(event.pos):
                            state.active_layer_idx = (state.active_layer_idx - 1) % len(state.layer_names)
                            scroll_y = 0
                        elif next_layer_rect.collidepoint(event.pos):
                            state.active_layer_idx = (state.active_layer_idx + 1) % len(state.layer_names)
                            scroll_y = 0
                        elif next_btn_rect.collidepoint(event.pos):
                            res = state.advance_position()
                            if res: board, attn_dict, is_black_to_move = res
                        elif prev_btn_rect.collidepoint(event.pos):
                            board, attn_dict, is_black_to_move = state.previous_position()
                        elif filter_btn_rect.collidepoint(event.pos):
                            show_filters = True

                    # Mouse Wheel Scrolling
                    elif event.button == 4:
                        scroll_y = min(0, scroll_y + 40)
                    elif event.button == 5:
                        scroll_y = max(min_scroll_y, scroll_y - 40)

            elif event.type == pygame.MOUSEBUTTONUP:
                active_handle = None

            elif event.type == pygame.MOUSEMOTION and show_filters and active_handle:
                p_code, handle_type = active_handle
                track_x, track_w = 680, 240
                rel_x = max(0, min(event.pos[0] - track_x, track_w))
                new_val = int(round((rel_x / track_w) * 16))
                
                if handle_type == 'min':
                    state.filters[p_code]['min'] = min(new_val, state.filters[p_code]['max'])
                else:
                    state.filters[p_code]['max'] = max(new_val, state.filters[p_code]['min'])

        if show_filters:
            draw_filter_dialog(screen, fonts, state.filters, dialog_reset_rect, dialog_close_rect, active_handle)
        else:
            draw_interface(screen, fonts, board, attn_dict, active_layer_name, 
                           state.selected_square, is_black_to_move, 
                           prev_btn_rect, next_btn_rect, filter_btn_rect, 
                           prev_layer_rect, next_layer_rect, scroll_y)

        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()