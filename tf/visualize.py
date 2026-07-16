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
# Intercept chunkparser to inject the true side_to_move (stm) indicator into the stream
original_convert = chunkparser.ChunkParserInner.convert_v7_to_tuple

def patched_convert_v7_to_tuple(self, content):
    res = original_convert(self, content)
    unpacked = chunkparser.v7b_struct.unpack(content)
    stm = unpacked[8] # Extract uint8 side_to_move_or_enpassant at index 8 directly
    return res + (struct.pack('B', stm),)

chunkparser.ChunkParserInner.convert_v7_to_tuple = patched_convert_v7_to_tuple


# --- CONFIGURATION & PIPELINE INITIALIZATION ---
def init_pipeline():
    config_path = os.environ.get("LEELA_CFG", "./configs/GGT3-shared.yaml")
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
    
    # Expand output types to 9 strings to capture our appended stm byte
    output_types = 9 * (tf.string,)
    
    def custom_parse_function(p1, p2, p3, p4, p5, p6, p7, p8, p9):
        tensors = parse_function(p1, p2, p3, p4, p5, p6, p7, p8)
        stm_tensor = tf.io.decode_raw(p9, tf.uint8)
        return tensors + (stm_tensor,)

    test_dataset = tf.data.Dataset.from_generator(parser.parse, output_types=output_types).map(custom_parse_function)
    
    tfprocess.init(test_dataset, test_dataset, None)
    tfprocess.restore()

    return iter(test_dataset), tfprocess, cfg['model'].get('gating_groups', 1), parser


# --- DATA EXTRACTION & FILTER ENGINE ---
class PositionState:
    def __init__(self, test_iter, tfprocess, groups):
        self.test_iter = test_iter
        self.tfprocess = tfprocess
        self.groups = groups
        self.batch_x = None
        self.batch_stm = None
        self.batch_size = 0
        self.current_idx = -1
        
        self.history = []  # Stores tuples of (board, gates_dict, is_black_to_move)
        self.history_pointer = -1

        self.filters = {
            'P': {'min': 0, 'max': 16},
            'N': {'min': 0, 'max': 16},
            'B': {'min': 0, 'max': 16},
            'R': {'min': 0, 'max': 16},
            'Q': {'min': 0, 'max': 16}
        }
        
        gating_out_layers = [l for l in self.tfprocess.model.layers if "layer_gating/out" in l.name]
        if not gating_out_layers:
            print("Error: Active architecture does not contain Gated Blocks ('G') matching layer_gating/out.")
            sys.exit(1)
            
        self.layer_names = sorted([l.name for l in gating_out_layers])
        self.active_layer_idx = 0
        
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
                print("Fetching next dataset batch array from stream...")
                try:
                    outputs = next(self.test_iter)
                except StopIteration:
                    print("End of dataset reached.")
                    return None
                
                self.batch_x = outputs[0]
                self.batch_stm = outputs[8].numpy() # Collect batched stm indices
                self.activations_dict = self.inspection_model(self.batch_x, training=False)
                self.batch_size = self.batch_x.shape[0]
                self.current_idx = 0
            else:
                self.current_idx += 1
            
            attempts += 1
            board, gates_dict, is_black_to_move = self.get_current_data()
            if self.matches_filter(board):
                self.history.append((board, gates_dict, is_black_to_move))
                self.history_pointer += 1
                return board, gates_dict, is_black_to_move
        
        print("Warning: 5000 positions scanned. No match found for current filters.")
        return None

    def previous_position(self):
        if self.history_pointer > 0:
            self.history_pointer -= 1
        return self.history[self.history_pointer]

    def get_current_data(self):
        stm_val = self.batch_stm[self.current_idx]
        is_black_to_move = (int(stm_val) == 1)

        # Planes 0-5 are ALWAYS current player, planes 6-11 are ALWAYS opponent
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

        gates_dict = {}
        for name in self.layer_names:
            raw_gating = self.activations_dict[name][self.current_idx]
            gating_weights = tf.sigmoid(raw_gating).numpy()
            if len(gating_weights.shape) == 1:
                gating_weights = np.reshape(gating_weights, [25, self.groups])
                
            group_gates = []
            for g in range(self.groups):
                gating_vector = gating_weights[:, g]
                group_gates.append(gating_vector.reshape(5, 5))
            gates_dict[name] = group_gates

        return board_matrix, gates_dict, is_black_to_move


# --- INTERFACE DESIGN GRAPHICS ENGINE ---
def draw_interface(screen, fonts, board, gates_dict, active_layer_name, is_black_to_move, prev_btn_rect, next_btn_rect, filter_btn_rect, prev_layer_rect, next_layer_rect, scroll_y, total_groups):
    screen.fill((30, 30, 35))

    # 1. Render Chess Board
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
                
                if piece.isupper(): # Capital letters represent White pieces globally
                    for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        shadow_text = fonts['piece'].render(glyph, True, (15, 15, 15))
                        screen.blit(shadow_text, shadow_text.get_rect(center=(cx + ox, cy + oy)))
                    main_text = fonts['piece'].render(glyph, True, (255, 255, 255))
                else: # Lowercase letters represent Black pieces globally
                    main_text = fonts['piece'].render(glyph, True, (20, 20, 20))
                    
                screen.blit(main_text, main_text.get_rect(center=(cx, cy)))

    # Rotate rank/file labels dynamically based on perspective
    if not is_black_to_move:
        rank_labels = [str(8 - i) for i in range(8)]
        file_labels = [chr(97 + i) for i in range(8)]
    else:
        rank_labels = [str(i + 1) for i in range(8)]
        file_labels = [chr(104 - i) for i in range(8)]

    # Draw Coordinate Labels
    for i in range(8):
        lbl = fonts['label'].render(rank_labels[i], True, (200, 200, 200))
        screen.blit(lbl, (board_x - 22, board_y + i * sq_size + 15))
        lbl = fonts['label'].render(file_labels[i], True, (200, 200, 200))
        screen.blit(lbl, (board_x + i * sq_size + 18, board_y + sq_size * 8 + 5))

    # 2. Render Layer Control Selection Headers
    mouse_pos = pygame.mouse.get_pos()
    for btn, color_base, text in [(prev_layer_rect, (60, 65, 70), "PREV LAYER"), 
                                  (next_layer_rect, (60, 65, 70), "NEXT LAYER")]:
        color = tuple(min(c + 25, 255) for c in color_base) if btn.collidepoint(mouse_pos) else color_base
        pygame.draw.rect(screen, color, btn, border_radius=4)
        lbl = fonts['sub_bold'].render(text, True, (230, 230, 235))
        screen.blit(lbl, lbl.get_rect(center=btn.center))

    clean_layer_label = active_layer_name.split("/layer_gating")[0]
    layer_lbl = fonts['main'].render(f"Block: {clean_layer_label}", True, (215, 225, 235))
    screen.blit(layer_lbl, (745, 55))

    # 3. Render Scrolling Gating Weight Heatmaps
    start_k_x, start_k_y = 480, 125
    cell_size = 32
    k_spacing_x, k_spacing_y = 230, 215

    clip_rect = pygame.Rect(465, 95, 520, 510)
    screen.set_clip(clip_rect)

    gates = gates_dict.get(active_layer_name, [np.zeros((5,5))] * total_groups)

    for idx, gating_matrix in enumerate(gates[:total_groups]):
        gx = idx % 2
        gy = idx // 2
        kx = start_k_x + gx * k_spacing_x
        ky = start_k_y + gy * k_spacing_y + scroll_y

        if ky + k_spacing_y < clip_rect.top or ky > clip_rect.bottom:
            continue

        title = fonts['sub_bold'].render(f"Group {idx} Gating Weights", True, (190, 210, 230))
        screen.blit(title, (kx, ky - 22))

        pygame.draw.rect(screen, (65, 65, 70), (kx - 2, ky - 2, cell_size * 5 + 4, cell_size * 5 + 4), 1)

        for r in range(5):
            for c in range(5):
                val = gating_matrix[r, c]
                intensity = int(val * 255)
                cell_color = (intensity, 55, 255 - intensity)
                cx = kx + c * cell_size
                cy = ky + r * cell_size
                pygame.draw.rect(screen, cell_color, (cx, cy, cell_size, cell_size))
                pygame.draw.rect(screen, (40, 40, 45), (cx, cy, cell_size, cell_size), 1)

                val_text = fonts['value'].render(f"{val:.2f}", True, (255, 255, 255) if intensity > 130 else (210, 210, 210))
                screen.blit(val_text, val_text.get_rect(center=(cx + cell_size//2, cy + cell_size//2)))

    screen.set_clip(None)

    # 4. Render Bottom Controls
    for btn, color_base, text in [(prev_btn_rect, (90, 95, 100), "PREV POS"), 
                                  (next_btn_rect, (50, 115, 70), "NEXT POS"), 
                                  (filter_btn_rect, (40, 90, 130), "FILTERS")]:
        color = tuple(min(c + 30, 255) for c in color_base) if btn.collidepoint(mouse_pos) else color_base
        pygame.draw.rect(screen, color, btn, border_radius=6)
        lbl = fonts['sub_bold'].render(text, True, (255, 255, 255))
        screen.blit(lbl, lbl.get_rect(center=btn.center))

    header = fonts['main'].render("LEELA CHESS INTERACTIVE GATING WEIGHTS PROFILER", True, (255, 255, 255))
    screen.blit(header, (50, 18))


def draw_filter_dialog(screen, fonts, filters, reset_btn_rect, close_btn_rect, active_handle):
    overlay = pygame.Surface((1000, 620), pygame.SRCALPHA)
    overlay.fill((10, 10, 15, 200))
    screen.blit(overlay, (0, 0))

    panel_rect = pygame.Rect(250, 100, 500, 420)
    pygame.draw.rect(screen, (40, 45, 50), panel_rect, border_radius=12)
    pygame.draw.rect(screen, (70, 80, 90), panel_rect, width=2, border_radius=12)

    title = fonts['main'].render("OCCURRENCE FILTERS CONFIGURATION", True, (255, 255, 255))
    screen.blit(title, title.get_rect(center=(500, 140)))

    piece_names = {'P': 'Pawns (P)', 'N': 'Knights (N)', 'B': 'Bishops (B)', 'R': 'Rooks (R)', 'Q': 'Queens (Q)'}
    
    track_x, track_w = 420, 240
    for idx, (p_code, name) in enumerate(piece_names.items()):
        row_y = 200 + idx * 50
        
        lbl = fonts['sub_bold'].render(name, True, (210, 220, 230))
        screen.blit(lbl, (280, row_y - 8))

        pygame.draw.line(screen, (70, 75, 80), (track_x, row_y), (track_x + track_w, row_y), 6)
        
        min_val, max_val = filters[p_code]['min'], filters[p_code]['max']
        min_x = track_x + int((min_val / 16) * track_w)
        max_x = track_x + int((max_val / 16) * track_w)

        pygame.draw.line(screen, (50, 130, 80), (min_x, row_y), (max_x, row_y), 6)

        pygame.draw.circle(screen, (220, 80, 80), (min_x, row_y), 8)
        pygame.draw.circle(screen, (80, 180, 120), (max_x, row_y), 8)

        val_lbl = fonts['label'].render(f"Min: {min_val}  Max: {max_val}", True, (200, 200, 200))
        screen.blit(val_lbl, (420, row_y + 10))

    # Render Reset and Close Buttons using the pre-defined layout rectangles
    mouse_pos = pygame.mouse.get_pos()
    
    # MODIFICATION: Draw the RESET button cleanly next to the Close button
    reset_color = (150, 80, 80) if reset_btn_rect.collidepoint(mouse_pos) else (120, 60, 60)
    pygame.draw.rect(screen, reset_color, reset_btn_rect, border_radius=6)
    r_lbl = fonts['sub_bold'].render("RESET", True, (255, 255, 255))
    screen.blit(r_lbl, r_lbl.get_rect(center=reset_btn_rect.center))
    
    # Draw Apply / Close Button
    btn_color = (70, 140, 90) if close_btn_rect.collidepoint(mouse_pos) else (50, 115, 70)
    pygame.draw.rect(screen, btn_color, close_btn_rect, border_radius=6)
    c_lbl = fonts['sub_bold'].render("APPLY & CLOSE", True, (255, 255, 255))
    screen.blit(c_lbl, c_lbl.get_rect(center=close_btn_rect.center))

# --- MAIN RUNNER LOOP ---
def main():
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    pygame.init()
    pygame.font.init()
    
    fallback_fonts = ['segoeuisymbol', 'dejavusans', 'freesans', 'arialunicodems', 'sans-serif']
    fonts = {
        'main': pygame.font.SysFont('sans-serif', 22, bold=True),
        'sub_bold': pygame.font.SysFont('sans-serif', 14, bold=True),
        'label': pygame.font.SysFont('monospace', 12, bold=True),
        'piece': pygame.font.SysFont(fallback_fonts, 38),  
        'value': pygame.font.SysFont('monospace', 11, bold=True)
    }

    print("Booting reproducible pipeline streaming engines...")
    test_iter, tfprocess, groups, parser = init_pipeline()
    
    state = PositionState(test_iter, tfprocess, groups)
    res = state.advance_position()
    board, gates_dict, is_black_to_move = res if res else ([["."]*8]*8, {}, False)

    screen = pygame.display.set_mode((1000, 620))
    pygame.display.set_caption("Leela Chess Gating Weights Interactive Inspector")
    
    # Bottom Panel Buttons
    prev_btn_rect = pygame.Rect(50, 520, 110, 40)
    next_btn_rect = pygame.Rect(175, 520, 110, 40)
    filter_btn_rect = pygame.Rect(300, 520, 110, 40)
    
    # Top Panel Layer Buttons
    prev_layer_rect = pygame.Rect(480, 50, 120, 32)
    next_layer_rect = pygame.Rect(610, 50, 120, 32)

    # MODIFICATION: Statically defined Dialog Control boundaries to avoid UnboundLocalErrors
    dialog_reset_rect = pygame.Rect(280, 460, 190, 40)
    dialog_close_rect = pygame.Rect(530, 460, 190, 40)
    
    scroll_y = 0
    max_rows = (groups + 1) // 2
    min_scroll_y = min(0, 510 - (max_rows * 215 + 40))

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
                        # MODIFICATION: Handle Reset & Apply button clicks dynamically in dialog state
                        if dialog_close_rect.collidepoint(event.pos):
                            show_filters = False
                            active_handle = None
                        elif dialog_reset_rect.collidepoint(event.pos):
                            for p_code in state.filters:
                                state.filters[p_code]['min'] = 0
                                state.filters[p_code]['max'] = 16
                        else:
                            track_x, track_w = 420, 240
                            for idx, p_code in enumerate(['P', 'N', 'B', 'R', 'Q']):
                                row_y = 200 + idx * 50
                                min_val = state.filters[p_code]['min']
                                max_val = state.filters[p_code]['max']
                                min_x = track_x + int((min_val / 16) * track_w)
                                max_x = track_x + int((max_val / 16) * track_w)
                                
                                dist_min = np.hypot(event.pos[0] - min_x, event.pos[1] - row_y)
                                dist_max = np.hypot(event.pos[0] - max_x, event.pos[1] - row_y)
                                
                                if dist_min < 10 or dist_max < 10:
                                    if min_val == max_val:
                                        if min_val == 0: active_handle = (p_code, 'max')
                                        elif min_val == 16: active_handle = (p_code, 'min')
                                        else: active_handle = (p_code, 'max')
                                    else:
                                        if dist_min < dist_max: active_handle = (p_code, 'min')
                                        else: active_handle = (p_code, 'max')
                                    break
                    else:
                        if prev_layer_rect.collidepoint(event.pos):
                            state.active_layer_idx = (state.active_layer_idx - 1) % len(state.layer_names)
                            scroll_y = 0 
                        elif next_layer_rect.collidepoint(event.pos):
                            state.active_layer_idx = (state.active_layer_idx + 1) % len(state.layer_names)
                            scroll_y = 0
                        
                        elif next_btn_rect.collidepoint(event.pos):
                            res = state.advance_position()
                            if res: board, gates_dict, is_black_to_move = res
                        elif prev_btn_rect.collidepoint(event.pos):
                            board, gates_dict, is_black_to_move = state.previous_position()
                        elif filter_btn_rect.collidepoint(event.pos):
                            show_filters = True
                
                elif event.type == pygame.MOUSEBUTTONDOWN and not show_filters:
                    if event.button == 4 and event.pos[0] > 450: 
                        scroll_y = min(0, scroll_y + 25)
                    elif event.button == 5 and event.pos[0] > 450: 
                        scroll_y = max(min_scroll_y, scroll_y - 25)

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
            draw_interface(screen, fonts, board, gates_dict, active_layer_name, is_black_to_move, prev_btn_rect, next_btn_rect, filter_btn_rect, prev_layer_rect, next_layer_rect, scroll_y, groups)

        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()