#!/usr/bin/env python3
import os
import yaml
import numpy as np
import pygame
import sys
import random
import struct
import gzip

# Import components from your training pipeline
import chunkparser
from train import get_latest_chunks, get_input_mode, game_number_for_name, identity_function

# --- CONFIGURATION & TEST SET CHUNKS PATHS ---
def init_pipeline():
    # Set default config fallback path to ./configs/GGT3-shared.yaml
    config_path = os.environ.get("LEELA_CFG", "./configs/GGT3-shared.yaml")
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at '{config_path}'. Please set the LEELA_CFG environment variable.")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Reconstruct test chunks using the dataset parameters
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

    return test_chunks, get_input_mode(cfg)


# --- DIRECT BINARY DATA PARSER ---
def read_chunk_positions(filename, expected_input_format):
    """Reads a single chunk (.gz file) sequentially, yielding board matrices and targets."""
    record_size = 8396
    dummy_prob = struct.pack("f", 1.0) + struct.pack("f", -1.0) * 1857
    BLOCK_RECORDS = 4000 
    BLOCK_SIZE = BLOCK_RECORDS * record_size

    try:
        with gzip.open(filename, "rb") as chunk_file:
            version = chunk_file.read(4)
            if version != b'\x07\x00\x00\x00':  # Check for valid V7 signature
                return
            chunk_file.seek(0)
            
            carry_over = b""
            while True:
                raw_block = chunk_file.read(BLOCK_SIZE)
                if not raw_block:
                    break
                    
                block_data = carry_over + raw_block
                n_block_records = len(block_data) // record_size
                if n_block_records == 0:
                    break
                    
                probs = [block_data[j + 8 : j + 8 + 1858 * 4] for j in range(0, n_block_records * record_size, record_size)]
                probs.extend(2 * [dummy_prob])

                all_plies = [struct.unpack("f", block_data[j + 8304 : j + 8308])[0] for j in range(0, n_block_records * record_size, record_size)]
                all_plies.extend([0.0, 0.0])
                
                is_eof = len(raw_block) < BLOCK_SIZE
                end_range = n_block_records if is_eof else (n_block_records - 2)
                
                for idx in range(end_range):
                    offset = idx * record_size
                    record = block_data[offset : offset + record_size]
                    
                    # Look-ahead boundary padding
                    if idx + 1 >= n_block_records:
                        record += dummy_prob + dummy_prob
                    elif idx + 2 >= n_block_records:
                        record += probs[idx + 1] + dummy_prob
                    else:
                        record += b"".join(probs[idx + 1 : idx + 3])
                    
                    try:
                        # Unpack V7B record fields using chunkparser definitions
                        unpacked = chunkparser.v7b_struct.unpack(record)
                    except Exception:
                        continue
                    
                    (ver, input_format, probs_raw, planes_raw, us_ooo, us_oo, them_ooo, them_oo,
                     stm, rule50_count, invariance_info, dep_result, root_q, best_q,
                     root_d, best_d, root_m, best_m, plies_left, result_q, result_d,
                     played_q, played_d, played_m, orig_q, orig_d, orig_m, visits,
                     played_idx, best_idx, pol_kld, st_q, st_d, opp_played_idx, next_played_idx,
                     f1, f2, f3, f4, f5, f6, f7, f8, opp_probs, next_probs) = unpacked
                    
                    is_black_to_move = (int(stm) == 1)
                    
                    # Map piece symbols from active player perspective
                    if not is_black_to_move:
                        piece_symbols = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
                    else:
                        piece_symbols = ["p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]

                    board_matrix = [["." for _ in range(8)] for _ in range(8)]
                    
                    # Read board representations from the first 12 binary planes (96 bytes)
                    first_12_planes = planes_raw[:96]
                    decoded_planes = np.unpackbits(np.frombuffer(first_12_planes, dtype=np.uint8)).reshape(12, 8, 8)
                    
                    for r in range(8):
                        for c in range(8):
                            for p_idx in range(12):
                                if decoded_planes[p_idx, r, c] == 1:
                                    board_matrix[r][c] = piece_symbols[p_idx]
                                    break
                    
                    # Map every single scalar target from the binary block
                    targets = {
                        "version": struct.unpack("i", ver)[0],
                        "input_format": input_format,
                        "side_to_move": "Black" if is_black_to_move else "White",
                        "rule50_count": rule50_count,
                        "invariance_info": invariance_info,
                        "dep_result": dep_result,
                        "plies_left": plies_left,
                        "result_q": result_q,
                        "result_d": result_d,
                        "root_q": root_q,
                        "root_d": root_d,
                        "root_m": root_m,
                        "best_q": best_q,
                        "best_d": best_d,
                        "best_m": best_m,
                        "st_q": st_q,
                        "st_d": st_d,
                        "orig_q": orig_q,
                        "orig_d": orig_d,
                        "orig_m": orig_m,
                        "played_q": played_q,
                        "played_d": played_d,
                        "played_m": played_m,
                        "visits": visits,
                        "played_idx": played_idx,
                        "best_idx": best_idx,
                        "opp_played_idx": opp_played_idx,
                        "next_played_idx": next_played_idx,
                        "pol_kld": pol_kld,
                        "castling_us_ooo": us_ooo,
                        "castling_us_oo": us_oo,
                        "castling_them_ooo": them_ooo,
                        "castling_them_oo": them_oo,
                        "f1": f1, "f2": f2, "f3": f3, "f4": f4,
                        "f5": f5, "f6": f6, "f7": f7, "f8": f8
                    }
                    
                    yield board_matrix, targets, is_black_to_move
                
                if not is_eof:
                    carry_over = block_data[(n_block_records - 2) * record_size : n_block_records * record_size]
                else:
                    break
    except Exception as e:
        print(f"Exception reading {filename}: {e}")
        return


# --- SEAMLESS POSITION HISTORY & CHUNK LOADER ---
class PositionState:
    def __init__(self, test_chunks, expected_input_format):
        self.test_chunks = test_chunks
        self.expected_input_format = expected_input_format
        
        # Bi-directional cache tracking to support clean POS backing-up
        self.history = []
        self.history_pointer = -1
        
        self.current_chunk_file = None
        self.chunk_positions = []
        self.chunk_idx = -1

    def load_random_chunk(self):
        if not self.test_chunks:
            print("Error: No test chunks found.")
            return False
            
        self.current_chunk_file = random.choice(self.test_chunks)
        print(f"Opening random chunk: {self.current_chunk_file}")
        
        # Read all positions inside this chunk sequentially
        self.chunk_positions = list(read_chunk_positions(self.current_chunk_file, self.expected_input_format))
        if not self.chunk_positions:
            return self.load_random_chunk() # Try another on failure
            
        self.chunk_idx = 0
        return True

    def advance_position(self):
        # 1. Travel forward in history if we backed up earlier
        if self.history_pointer < len(self.history) - 1:
            self.history_pointer += 1
            return self.history[self.history_pointer]
            
        # 2. First execution sequence
        if self.current_chunk_file is None:
            if not self.load_random_chunk():
                return None
                
        # 3. Read the next successive element from the open chunk
        elif self.chunk_idx < len(self.chunk_positions) - 1:
            self.chunk_idx += 1
            
        # 4. End of chunk: Open a new random chunk file
        else:
            if not self.load_random_chunk():
                return None
                
        board, targets, is_black_to_move = self.chunk_positions[self.chunk_idx]
        state_tuple = (self.current_chunk_file, self.chunk_idx, len(self.chunk_positions), board, targets, is_black_to_move)
        
        self.history.append(state_tuple)
        self.history_pointer += 1
        return state_tuple

    def previous_position(self):
        if self.history_pointer > 0:
            self.history_pointer -= 1
            
        # Resynchronize current chunk pointer indices on backsteps
        if self.history_pointer >= 0:
            hist_state = self.history[self.history_pointer]
            self.current_chunk_file = hist_state[0]
            self.chunk_idx = hist_state[1]
            self.chunk_positions = list(read_chunk_positions(self.current_chunk_file, self.expected_input_format))
            
        return self.history[self.history_pointer]


# --- INTERFACE DESIGN ENGINE ---
def draw_target_grid(screen, fonts, targets, start_x, start_y):
    """Renders scalar targets beautifully aligned in a dense, scannable two-column table."""
    keys = list(targets.keys())
    col_size = (len(keys) + 1) // 2
    
    col1_keys = keys[:col_size]
    col2_keys = keys[col_size:]
    line_h = 24
    
    # Column 1
    for idx, key in enumerate(col1_keys):
        val = targets[key]
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        
        lbl = fonts['sub_bold'].render(f"{key}:", True, (180, 200, 220))
        val_lbl = fonts['value'].render(val_str, True, (255, 255, 255))
        
        screen.blit(lbl, (start_x, start_y + idx * line_h))
        screen.blit(val_lbl, (start_x + 130, start_y + idx * line_h))
        
    # Column 2
    for idx, key in enumerate(col2_keys):
        val = targets[key]
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        
        lbl = fonts['sub_bold'].render(f"{key}:", True, (180, 200, 220))
        val_lbl = fonts['value'].render(val_str, True, (255, 255, 255))
        
        screen.blit(lbl, (start_x + 260, start_y + idx * line_h))
        screen.blit(val_lbl, (start_x + 390, start_y + idx * line_h))


def draw_interface(screen, fonts, board, targets, is_black_to_move, filename, pos_idx, total_pos, prev_btn_rect, next_btn_rect):
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
                
                if piece.isupper():
                    for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        shadow_text = fonts['piece'].render(glyph, True, (15, 15, 15))
                        screen.blit(shadow_text, shadow_text.get_rect(center=(cx + ox, cy + oy)))
                    main_text = fonts['piece'].render(glyph, True, (255, 255, 255))
                else:
                    main_text = fonts['piece'].render(glyph, True, (20, 20, 20))
                    
                screen.blit(main_text, main_text.get_rect(center=(cx, cy)))

    # Dynamically rotate board rank and file coordinate text labels
    if not is_black_to_move:
        rank_labels = [str(8 - i) for i in range(8)]
        file_labels = [chr(97 + i) for i in range(8)]
    else:
        rank_labels = [str(i + 1) for i in range(8)]
        file_labels = [chr(104 - i) for i in range(8)]

    for i in range(8):
        lbl = fonts['label'].render(rank_labels[i], True, (200, 200, 200))
        screen.blit(lbl, (board_x - 22, board_y + i * sq_size + 15))
        lbl = fonts['label'].render(file_labels[i], True, (200, 200, 200))
        screen.blit(lbl, (board_x + i * sq_size + 18, board_y + sq_size * 8 + 5))

    # 2. Render Left Bottom Metadata labels
    short_filename = os.path.basename(filename)
    file_lbl = fonts['sub_bold'].render(f"Chunk File: {short_filename}", True, (200, 200, 200))
    pos_lbl = fonts['sub_bold'].render(f"Successive Position: {pos_idx + 1} / {total_pos}", True, (200, 200, 200))
    screen.blit(file_lbl, (50, 440))
    screen.blit(pos_lbl, (50, 470))

    # 3. Render Navigation Control Buttons (No rendering-breaking text symbols)
    mouse_pos = pygame.mouse.get_pos()
    for btn, color_base, text in [(prev_btn_rect, (90, 95, 100), "PREV POS"), 
                                  (next_btn_rect, (50, 115, 70), "NEXT POS")]:
        color = tuple(min(c + 30, 255) for c in color_base) if btn.collidepoint(mouse_pos) else color_base
        pygame.draw.rect(screen, color, btn, border_radius=6)
        lbl = fonts['sub_bold'].render(text, True, (255, 255, 255))
        screen.blit(lbl, lbl.get_rect(center=btn.center))

    # 4. Render Targets Section
    target_header = fonts['main'].render("SCALAR TARGETS INSPECTION", True, (215, 225, 235))
    screen.blit(target_header, (480, 55))
    draw_target_grid(screen, fonts, targets, 480, 95)

    header = fonts['main'].render("LEELA CHESS TRAINING DATA INSPECTOR", True, (255, 255, 255))
    screen.blit(header, (50, 18))


# --- MAIN PROGRAM LOOP ---
def main():
    random.seed(42)
    np.random.seed(42)

    pygame.init()
    pygame.font.init()
    
    fallback_fonts = ['segoeuisymbol', 'dejavusans', 'freesans', 'arialunicodems', 'sans-serif']
    fonts = {
        'main': pygame.font.SysFont('sans-serif', 22, bold=True),
        'sub_bold': pygame.font.SysFont('sans-serif', 14, bold=True),
        'label': pygame.font.SysFont('monospace', 12, bold=True),
        'piece': pygame.font.SysFont(fallback_fonts, 38),  
        'value': pygame.font.SysFont('monospace', 12, bold=True)
    }

    print("Booting chunk loader pipeline...")
    test_chunks, expected_input_format = init_pipeline()
    
    state = PositionState(test_chunks, expected_input_format)
    res = state.advance_position()
    
    if not res:
        print("Error: Could not load position state.")
        sys.exit(1)
        
    filename, pos_idx, total_pos, board, targets, is_black_to_move = res

    screen = pygame.display.set_mode((1000, 620))
    pygame.display.set_caption("Leela Chess Test Set Inspector")
    
    # Expanded Navigation Buttons
    prev_btn_rect = pygame.Rect(50, 515, 150, 40)
    next_btn_rect = pygame.Rect(215, 515, 150, 40)
    
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if next_btn_rect.collidepoint(event.pos):
                        res = state.advance_position()
                        if res:
                            filename, pos_idx, total_pos, board, targets, is_black_to_move = res
                    elif prev_btn_rect.collidepoint(event.pos):
                        res = state.previous_position()
                        if res:
                            filename, pos_idx, total_pos, board, targets, is_black_to_move = res

        draw_interface(screen, fonts, board, targets, is_black_to_move, filename, pos_idx, total_pos, prev_btn_rect, next_btn_rect)
        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()