#!/usr/bin/env python3
import os
import yaml
import numpy as np
import tensorflow as tf
import pygame
import sys
import random
import struct

# Import components from your training pipeline
import chunkparser
from train import get_latest_chunks, get_input_mode, game_number_for_name, identity_function, SKIP
from chunkparser import ChunkParser
from chunkparsefunc import parse_function
from tfprocess import TFProcess

# --- RUNTIME MONKEYPATCH FOR SIDE-TO-MOVE EXTRACTION ---
original_convert = chunkparser.ChunkParserInner.convert_v7_to_tuple

def patched_convert_v7_to_tuple(self, content):
    res = original_convert(self, content)
    unpacked = chunkparser.v7b_struct.unpack(content)
    stm = unpacked[8] # Extract uint8 side_to_move_or_enpassant at index 8 directly
    return res + (struct.pack('B', stm),)

chunkparser.ChunkParserInner.convert_v7_to_tuple = patched_convert_v7_to_tuple


# --- CONFIGURATION & PIPELINE INITIALIZATION ---
def init_pipeline():
    config_path = os.environ.get("LEELA_CFG", "./configs/MMTMMTMMT-cuda.yaml")
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at '{config_path}'. Please set the LEELA_CFG environment variable.")
        sys.exit(1)

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

    total_batch_size = cfg["training"]["batch_size"]
    batch_splits = cfg["training"].get("num_batch_splits", 1)
    split_batch_size = total_batch_size // batch_splits
    test_shuffle_size = int(cfg["training"]["shuffle_size"] * (1.0 - train_ratio))

    parser = ChunkParser(
        test_chunks, get_input_mode(cfg), shuffle_size=test_shuffle_size,
        sample=SKIP, batch_size=split_batch_size, workers=cfg["dataset"].get("test_workers", 0)
    )

    tfprocess = TFProcess(cfg)
    output_types = 9 * (tf.string,)
    
    def custom_parse_function(p1, p2, p3, p4, p5, p6, p7, p8, p9):
        tensors = parse_function(p1, p2, p3, p4, p5, p6, p7, p8)
        stm_tensor = tf.io.decode_raw(p9, tf.uint8)
        return tensors + (stm_tensor,)

    test_dataset = tf.data.Dataset.from_generator(parser.parse, output_types=output_types).map(custom_parse_function)
    
    tfprocess.init(test_dataset, test_dataset, None)
    tfprocess.restore()

    return iter(test_dataset), tfprocess, parser


# --- SPATIAL RELEVANCE MAP EXTRACTION ENGINE ---
class PositionState:
    def __init__(self, test_iter, tfprocess):
        self.test_iter = test_iter
        self.tfprocess = tfprocess
        self.batch_x = None
        self.batch_stm = None
        self.batch_size = 0
        self.current_idx = -1
        
        self.history = []  # Stores tuples of (board, relevance_profiles, is_black_to_move)
        self.history_pointer = -1

        self.filters = {
            'P': {'min': 0, 'max': 16}, 'N': {'min': 0, 'max': 16},
            'B': {'min': 0, 'max': 16}, 'R': {'min': 0, 'max': 16},
            'Q': {'min': 0, 'max': 16}
        }
        
        self.dw_layers = [l for l in self.tfprocess.model.layers if "2/conv2d" in l.name]
        if not self.dw_layers:
            print("Error: Active model architecture does not contain any FusedChessDepthwise operations.")
            sys.exit(1)
            
        self.layer_names = sorted([l.name for l in self.dw_layers])
        self.active_layer_idx = 0
        
        # Target the OUTPUT of the depthwise operation to capture post-convolution states
        self.inspection_model = tf.keras.Model(
            inputs=self.tfprocess.model.input,
            outputs={name: self.tfprocess.model.get_layer(name).output for name in self.layer_names}
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
                print("Streaming next raw dataset block...")
                try:
                    outputs = next(self.test_iter)
                except StopIteration:
                    return None
                
                self.batch_x = outputs[0]
                self.batch_stm = outputs[8].numpy()
                self.raw_activations = self.inspection_model(self.batch_x, training=False)
                self.batch_size = self.batch_x.shape[0]
                self.current_idx = 0
            else:
                self.current_idx += 1
            
            attempts += 1
            board, relevance_profiles, is_black_to_move = self.get_current_data()
            if self.matches_filter(board):
                self.history.append((board, relevance_profiles, is_black_to_move))
                self.history_pointer += 1
                return board, relevance_profiles, is_black_to_move
        return None

    def previous_position(self):
        if self.history_pointer > 0:
            self.history_pointer -= 1
        return self.history[self.history_pointer]

    def get_current_data(self):
        stm_val = self.batch_stm[self.current_idx]
        is_black_to_move = (int(stm_val) == 1)

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

        relevance_profiles = {}
        for name in self.layer_names:
            layer_obj = self.tfprocess.model.get_layer(name)
            desc = layer_obj.get_mask_descriptor() # [Rook_C, Bishop_C, Knight_C]
            r_c, b_c, k_c = desc[0], desc[1], desc[2]
            
            # Reshape intermediate activations back to spatial layout [8, 8, Channels]
            pos_acts = self.raw_activations[name][self.current_idx].numpy()
            if len(pos_acts.shape) == 2:
                pos_acts = pos_acts.reshape(8, 8, -1)
            pos_acts = np.abs(pos_acts)
            
            # Compute spatial relevance map (Mean Absolute Activation per square)
            r_map = np.mean(pos_acts[:, :, 0 : r_c], axis=-1) if r_c > 0 else np.zeros((8, 8))
            b_map = np.mean(pos_acts[:, :, r_c : r_c + b_c], axis=-1) if b_c > 0 else np.zeros((8, 8))
            k_map = np.mean(pos_acts[:, :, r_c + b_c : r_c + b_c + k_c], axis=-1) if k_c > 0 else np.zeros((8, 8))
            
            def normalize_map(m):
                mx = np.max(m)
                return m / (mx + 1e-8), float(mx)

            norm_r, max_r = normalize_map(r_map)
            norm_b, max_b = normalize_map(b_map)
            norm_k, max_k = normalize_map(k_map)

            relevance_profiles[name] = {
                'desc': desc,
                'Rook': norm_r, 'Bishop': norm_b, 'Knight': norm_k,
                'max_raw': [max_r, max_b, max_k]
            }

        return board_matrix, relevance_profiles, is_black_to_move


# --- INTERFACE GRAPHICS & MINI-HEATMAP RENDERERS ---
def draw_mini_heatmap(screen, fonts, board, heatmap, start_x, start_y, sq_size, title, max_raw, color_mask):
    """Draws a dedicated small spatial board showing feature intensity maps."""
    lbl = fonts['sub_bold'].render(f"{title} (Max Abs: {max_raw:.2f})", True, (225, 230, 240))
    screen.blit(lbl, (start_x, start_y - 18))
    
    pygame.draw.rect(screen, (55, 55, 60), (start_x - 2, start_y - 2, sq_size * 8 + 4, sq_size * 8 + 4), 2)
    
    unicode_pieces = {'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
                      'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'}

    for r in range(8):
        for c in range(8):
            base_color = 45 if (r + c) % 2 == 0 else 25
            val = heatmap[r, c] # Normalized 0.0 -> 1.0
            
            # Blend background board square into the target theme color channel bounds
            r_c = int(base_color + val * (color_mask[0] - base_color))
            g_c = int(base_color + val * (color_mask[1] - base_color))
            b_c = int(base_color + val * (color_mask[2] - base_color))
            
            rect = (start_x + c * sq_size, start_y + r * sq_size, sq_size, sq_size)
            pygame.draw.rect(screen, (r_c, g_c, b_c), rect)
            pygame.draw.rect(screen, (20, 20, 25), rect, 1) # Wireframe gridlines
            
            piece = board[r][c]
            if piece != ".":
                glyph = unicode_pieces.get(piece, piece)
                cx = start_x + c * sq_size + sq_size // 2
                cy = start_y + r * sq_size + sq_size // 2
                
                p_color = (240, 240, 240) if val > 0.4 else (130, 135, 140)
                p_text = fonts['mini_piece'].render(glyph, True, p_color)
                screen.blit(p_text, p_text.get_rect(center=(cx, cy)))


def draw_interface(screen, fonts, board, profiles, active_layer_name, is_black_to_move, prev_btn_rect, next_btn_rect, filter_btn_rect, prev_layer_rect, next_layer_rect):
    screen.fill((30, 30, 35))

    # 1. Render Reference Main Board
    board_x, board_y, sq_size = 50, 70, 45
    pygame.draw.rect(screen, (50, 50, 55), (board_x - 4, board_y - 4, sq_size*8 + 8, sq_size*8 + 8), 3)
    
    unicode_pieces = {'K': '♚', 'Q': '♛', 'R': '♜', 'B': '♝', 'N': '♞', 'P': '♟',
                      'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'}

    for r in range(8):
        for c in range(8):
            color = (238, 238, 210) if (r + c) % 2 == 0 else (118, 150, 86)
            rect = (board_x + c * sq_size, board_y + r * sq_size, sq_size, sq_size)
            pygame.draw.rect(screen, color, rect)
            
            piece = board[r][c]
            if piece != ".":
                glyph = unicode_pieces.get(piece, piece)
                cx = board_x + c * sq_size + sq_size // 2
                cy = board_y + r * sq_size + sq_size // 2
                if piece.isupper():
                    for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        shadow_text = fonts['piece'].render(glyph, True, (15, 15, 15))
                        screen.blit(shadow_text, shadow_text.get_rect(center=(cx + ox, cy + oy)))
                    main_text = fonts['piece'].render(glyph, True, (255, 255, 255))
                else:
                    main_text = fonts['piece'].render(glyph, True, (20, 20, 20))
                screen.blit(main_text, main_text.get_rect(center=(cx, cy)))

    if not is_black_to_move:
        rank_labels, file_labels = [str(8 - i) for i in range(8)], [chr(97 + i) for i in range(8)]
    else:
        rank_labels, file_labels = [str(i + 1) for i in range(8)], [chr(104 - i) for i in range(8)]

    for i in range(8):
        screen.blit(fonts['label'].render(rank_labels[i], True, (200, 200, 200)), (board_x - 22, board_y + i * sq_size + 15))
        screen.blit(fonts['label'].render(file_labels[i], True, (200, 200, 200)), (board_x + i * sq_size + 18, board_y + sq_size * 8 + 5))

    # 2. Layer Selection
    mouse_pos = pygame.mouse.get_pos()
    for btn, color_base, text in [(prev_layer_rect, (60, 65, 70), "PREV LAYER"), 
                                  (next_layer_rect, (60, 65, 70), "NEXT LAYER")]:
        color = tuple(min(c + 25, 255) for c in color_base) if btn.collidepoint(mouse_pos) else color_base
        pygame.draw.rect(screen, color, btn, border_radius=4)
        screen.blit(fonts['sub_bold'].render(text, True, (230, 230, 235)), fonts['sub_bold'].render(text, True, (230, 230, 235)).get_rect(center=btn.center))

    clean_layer_label = active_layer_name.split("/2/conv2d")[0]
    screen.blit(fonts['main'].render(f"Block: {clean_layer_label}", True, (215, 225, 235)), (720, 55))

    # 3. Render Three Concurrent Group Replicated Spatial Heatmaps
    screen.blit(fonts['main'].render("POST-DW CHANNEL SPATIAL RELEVANCE MAPS", True, (220, 225, 235)), (480, 95))
    layer_profile = profiles.get(active_layer_name, None)
    
    if layer_profile:
        # Top Row: Rook & Bishop Spatial Views
        draw_mini_heatmap(screen, fonts, board, layer_profile['Rook'], 490, 140, 24, "Rook Masked Channels", layer_profile['max_raw'][0], (225, 85, 85))
        draw_mini_heatmap(screen, fonts, board, layer_profile['Bishop'], 745, 140, 24, "Bishop Masked Channels", layer_profile['max_raw'][1], (85, 135, 225))
        
        # Bottom Row: Knight Spatial View + Text Layout Panel
        draw_mini_heatmap(screen, fonts, board, layer_profile['Knight'], 490, 375, 24, "Knight Masked Channels", layer_profile['max_raw'][2], (85, 185, 125))
        
        # Bottom Information Box Container
        pygame.draw.rect(screen, (45, 45, 50), (745, 375, 192, 192), border_radius=6)
        pygame.draw.rect(screen, (60, 65, 70), (745, 375, 192, 192), width=1, border_radius=6)
        
        info_title = fonts['sub_bold'].render("Channel Group Sizes", True, (200, 210, 220))
        screen.blit(info_title, (760, 390))
        
        desc = layer_profile['desc']
        screen.blit(fonts['label'].render(f"Rook Channels   : {desc[0]}", True, (230, 230, 235)), (760, 425))
        screen.blit(fonts['label'].render(f"Bishop Channels : {desc[1]}", True, (230, 230, 235)), (760, 455))
        screen.blit(fonts['label'].render(f"Knight Channels : {desc[2]}", True, (230, 230, 235)), (760, 485))

    # 4. Bottom Navigation Toolbar
    for btn, color_base, text in [(prev_btn_rect, (90, 95, 100), "PREV POS"), 
                                  (next_btn_rect, (50, 115, 70), "NEXT POS"), 
                                  (filter_btn_rect, (40, 90, 130), "FILTERS")]:
        color = tuple(min(c + 30, 255) for c in color_base) if btn.collidepoint(mouse_pos) else color_base
        pygame.draw.rect(screen, color, btn, border_radius=6)
        screen.blit(fonts['sub_bold'].render(text, True, (255, 255, 255)), fonts['sub_bold'].render(text, True, (255, 255, 255)).get_rect(center=btn.center))

    screen.blit(fonts['main'].render("LEELA CHESS INTERACTIVE CHANNELS ANALYZER", True, (255, 255, 255)), (50, 18))


def draw_filter_dialog(screen, fonts, filters, reset_btn_rect, close_btn_rect, active_handle):
    overlay = pygame.Surface((1000, 620), pygame.SRCALPHA)
    overlay.fill((10, 10, 15, 200))
    screen.blit(overlay, (0, 0))

    panel_rect = pygame.Rect(250, 100, 500, 420)
    pygame.draw.rect(screen, (40, 45, 50), panel_rect, border_radius=12)
    pygame.draw.rect(screen, (70, 80, 90), panel_rect, width=2, border_radius=12)

    screen.blit(fonts['main'].render("OCCURRENCE FILTERS CONFIGURATION", True, (255, 255, 255)), fonts['main'].render("OCCURRENCE FILTERS CONFIGURATION", True, (255, 255, 255)).get_rect(center=(500, 140)))

    piece_names = {'P': 'Pawns (P)', 'N': 'Knights (N)', 'B': 'Bishops (B)', 'R': 'Rooks (R)', 'Q': 'Queens (Q)'}
    track_x, track_w = 420, 240
    for idx, (p_code, name) in enumerate(piece_names.items()):
        row_y = 200 + idx * 50
        screen.blit(fonts['sub_bold'].render(name, True, (210, 220, 230)), (280, row_y - 8))
        pygame.draw.line(screen, (70, 75, 80), (track_x, row_y), (track_x + track_w, row_y), 6)
        
        min_val, max_val = filters[p_code]['min'], filters[p_code]['max']
        min_x = track_x + int((min_val / 16) * track_w)
        max_x = track_x + int((max_val / 16) * track_w)

        pygame.draw.line(screen, (50, 130, 80), (min_x, row_y), (max_x, row_y), 6)
        pygame.draw.circle(screen, (220, 80, 80), (min_x, row_y), 8)
        pygame.draw.circle(screen, (80, 180, 120), (max_x, row_y), 8)
        screen.blit(fonts['label'].render(f"Min: {min_val}  Max: {max_val}", True, (200, 200, 200)), (420, row_y + 10))

    mouse_pos = pygame.mouse.get_pos()
    pygame.draw.rect(screen, (150, 80, 80) if reset_btn_rect.collidepoint(mouse_pos) else (120, 60, 60), reset_btn_rect, border_radius=6)
    screen.blit(fonts['sub_bold'].render("RESET", True, (255, 255, 255)), fonts['sub_bold'].render("RESET", True, (255, 255, 255)).get_rect(center=reset_btn_rect.center))
    
    pygame.draw.rect(screen, (70, 140, 90) if close_btn_rect.collidepoint(mouse_pos) else (50, 115, 70), close_btn_rect, border_radius=6)
    screen.blit(fonts['sub_bold'].render("APPLY & CLOSE", True, (255, 255, 255)), fonts['sub_bold'].render("APPLY & CLOSE", True, (255, 255, 255)).get_rect(center=close_btn_rect.center))

# --- MAIN RUNNER LOOP ---
def main():
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    pygame.init()
    pygame.font.init()
    
    fallback_fonts = ['segoeuisymbol', 'dejavusans', 'freesans', 'arialunicodems', 'sans-serif']
    fonts = {
        'main': pygame.font.SysFont('sans-serif', 21, bold=True),
        'sub_bold': pygame.font.SysFont('sans-serif', 14, bold=True),
        'label': pygame.font.SysFont('monospace', 12, bold=True),
        'piece': pygame.font.SysFont(fallback_fonts, 38),  
        'mini_piece': pygame.font.SysFont(fallback_fonts, 18),
        'value': pygame.font.SysFont('monospace', 11, bold=True)
    }

    print("Booting specialized attribution analysis tracking stream...")
    test_iter, tfprocess, parser = init_pipeline()
    
    state = PositionState(test_iter, tfprocess)
    res = state.advance_position()
    board, profiles, is_black_to_move = res if res else ([["."]*8]*8, {}, False)

    screen = pygame.display.set_mode((1000, 620))
    pygame.display.set_caption("Leela Chess MobileNet Spatial Relevance Analyzer")
    
    prev_btn_rect = pygame.Rect(50, 520, 110, 40)
    next_btn_rect = pygame.Rect(175, 520, 110, 40)
    filter_btn_rect = pygame.Rect(300, 520, 110, 40)
    
    prev_layer_rect = pygame.Rect(480, 50, 110, 32)
    next_layer_rect = pygame.Rect(600, 50, 110, 32)

    dialog_reset_rect = pygame.Rect(280, 460, 190, 40)
    dialog_close_rect = pygame.Rect(530, 460, 190, 40)
    
    show_filters = False
    active_handle = None 
    clock = pygame.time.Clock()

    while True:
        active_layer_name = state.layer_names[state.active_layer_idx]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                try: parser.shutdown()
                except IndexError: pass
                pygame.quit(); sys.exit()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if show_filters:
                        if dialog_close_rect.collidepoint(event.pos):
                            show_filters = False
                            active_handle = None
                        elif dialog_reset_rect.collidepoint(event.pos):
                            for p_code in state.filters:
                                state.filters[p_code]['min'], state.filters[p_code]['max'] = 0, 16
                        else:
                            track_x, track_w = 420, 240
                            for idx, p_code in enumerate(['P', 'N', 'B', 'R', 'Q']):
                                row_y = 200 + idx * 50
                                min_val, max_val = state.filters[p_code]['min'], state.filters[p_code]['max']
                                min_x = track_x + int((min_val / 16) * track_w)
                                max_x = track_x + int((max_val / 16) * track_w)
                                
                                dist_min = np.hypot(event.pos[0] - min_x, event.pos[1] - row_y)
                                dist_max = np.hypot(event.pos[0] - max_x, event.pos[1] - row_y)
                                
                                if dist_min < 10 or dist_max < 10:
                                    if min_val == max_val:
                                        active_handle = (p_code, 'min' if min_val == 16 else 'max')
                                    else:
                                        active_handle = (p_code, 'min' if dist_min < dist_max else 'max')
                                    break
                    else:
                        if prev_layer_rect.collidepoint(event.pos):
                            state.active_layer_idx = (state.active_layer_idx - 1) % len(state.layer_names)
                        elif next_layer_rect.collidepoint(event.pos):
                            state.active_layer_idx = (state.active_layer_idx + 1) % len(state.layer_names)
                        elif next_btn_rect.collidepoint(event.pos):
                            res = state.advance_position()
                            if res: board, profiles, is_black_to_move = res
                        elif prev_btn_rect.collidepoint(event.pos):
                            board, profiles, is_black_to_move = state.previous_position()
                        elif filter_btn_rect.collidepoint(event.pos):
                            show_filters = True

            elif event.type == pygame.MOUSEBUTTONUP:
                active_handle = None

            elif event.type == pygame.MOUSEMOTION and show_filters and active_handle:
                p_code, handle_type = active_handle
                track_x, track_w = 420, 240
                rel_x = max(0, min(event.pos[0] - track_x, track_w))
                new_val = int(round((rel_x / track_w) * 16))
                
                if handle_type == 'min':
                    state.filters[p_code]['min'] = min(new_val, state.filters[p_code]['max'])
                else:
                    state.filters[p_code]['max'] = max(new_val, state.filters[p_code]['min'])

        if show_filters:
            draw_filter_dialog(screen, fonts, state.filters, dialog_reset_rect, dialog_close_rect, active_handle)
        else:
            draw_interface(screen, fonts, board, profiles, active_layer_name, is_black_to_move, prev_btn_rect, next_btn_rect, filter_btn_rect, prev_layer_rect, next_layer_rect)

        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()