###################################
#
#   ai model(1dan) with gtp
#       by kai-sheng Huang, yi-yun Lee
#   National Dong Hwa University - NDHU
#
###################################
#
#   edit from WALLY by Jonathan K. Millen
#     (reconstruction by CMK)
#   https://github.com/maksimKorzh/wally/blob/main/tutorials/wally_07.py
#
###################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import random
import numpy as np
import queue
from queue import Queue
from keras.models import load_model
import tkinter as tk
import time
import threading
from top5 import top5


VERSION = '1.0'
FOLDER_PATH = sys.path[0] + '/models'

top5_thread = None
print('start')

###################################
#
#          Piece encoding
#
###################################
#
# 0000 => 0    empty sqare
# 0001 => 1    black stone
# 0010 => 2    white stone
# 0100 => 4    stone marker
# 0111 => 7    offboard square
# 1000 => 8    liberty marker
#
# 0101 => 5    black stone marked
# 0110 => 6    white stone marked
#
###################################

# 宣告變數
player_board = 0  # 模型
oppnent_board = 1  # 玩家
player_air_1 = 2
player_air_2 = 3
player_air_3 = 4
player_air_4 = 5
oppnent_air_1 = 6
oppnent_air_2 = 7
oppnent_air_3 = 8
oppnent_air_4 = 9
empty_board = 10
last_1 = 11
last_8 = 18
round_7 = 19
round_5 = 22
round_3 = 25

x = np.zeros((19, 19, 19))  # feature map
x[:, :, empty_board] = 1  # 空位盤面全設1
# model_style_fm = np.zeros((19, 19, 28))
# player_style_fm = np.zeros((19, 19, 28))
dan_model = None
# style_model = load_model(FOLDER_PATH + "/model_style_v2_b32_f256_l100_pempty_02.h5")
top_20_move = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
# sty_cnt = [[0, 0, 0], [0, 0, 0]]

chars = 'abcdefghijklmnopqrs'
numbertochar = {k:v for k,v in enumerate(chars)}

# 9x9 GO ban
board_9x9 = [
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
]

# 9x9 coordinates
coords_9x9 = [
    'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX',
    'XX', 'A9', 'B9', 'C9', 'D9', 'E9', 'F9', 'G9', 'H9', 'J9', 'XX',
    'XX', 'A8', 'B8', 'C8', 'D8', 'E8', 'F8', 'G8', 'H8', 'J8', 'XX',
    'XX', 'A7', 'B7', 'C7', 'D7', 'E7', 'F7', 'G7', 'H7', 'J7', 'XX',
    'XX', 'A6', 'B6', 'C6', 'D6', 'E6', 'F6', 'G6', 'H6', 'J6', 'XX',
    'XX', 'A5', 'B5', 'C5', 'D5', 'E5', 'F5', 'G5', 'H5', 'J5', 'XX',
    'XX', 'A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4', 'H4', 'J4', 'XX',
    'XX', 'A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3', 'J3', 'XX',
    'XX', 'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'J2', 'XX',
    'XX', 'A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'J1', 'XX',
    'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX'
]

# 13x13 GO ban
board_13x13 = [
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
]

# 13x13 coordinates
coords_13x13 = [
    'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX',
    'XX', 'A13','B13','C13','D13','E13','F13','G13','H13','J13','K13','L13','M13','N13','XX',
    'XX', 'A12','B12','C12','D12','E12','F12','G12','H12','J12','K12','L12','M12','N12','XX',
    'XX', 'A11','B11','C11','D11','E11','F11','G11','H11','J11','K11','L11','M11','N11','XX',
    'XX', 'A10','B10','C10','D10','E10','F10','G10','H10','J10','K10','L10','M10','N10','XX',
    'XX', 'A9', 'B9', 'C9', 'D9', 'E9', 'F9', 'G9', 'H9', 'J9', 'K9', 'L9', 'M9', 'N9', 'XX',
    'XX', 'A8', 'B8', 'C8', 'D8', 'E8', 'F8', 'G8', 'H8', 'J8', 'K8', 'L8', 'M8', 'N8', 'XX',
    'XX', 'A7', 'B7', 'C7', 'D7', 'E7', 'F7', 'G7', 'H7', 'J7', 'K7', 'L7', 'M7', 'N7', 'XX',
    'XX', 'A6', 'B6', 'C6', 'D6', 'E6', 'F6', 'G6', 'H6', 'J6', 'K6', 'L6', 'M6', 'N6', 'XX',
    'XX', 'A5', 'B5', 'C5', 'D5', 'E5', 'F5', 'G5', 'H5', 'J5', 'K5', 'L5', 'M5', 'N5', 'XX',
    'XX', 'A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4', 'H4', 'J4', 'K4', 'L4', 'M4', 'N4', 'XX',
    'XX', 'A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3', 'J3', 'K3', 'L3', 'M3', 'N3', 'XX',
    'XX', 'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'J2', 'K2', 'L2', 'M2', 'N2', 'XX',
    'XX', 'A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'J1', 'K1', 'L1', 'M1', 'N1', 'XX',
    'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX'
]

# 19x19 GO ban
board_19x19 = [
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
]

# 19x19 coordinates
coords_19x19 = [
    'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX',
    'XX', 'A19','B19','C19','D19','E19','F19','G19','H19','J19','K19','L19','M19','N19','O19','P19','Q19','R19','S19','T19','XX',
    'XX', 'A18','B18','C18','D18','E18','F18','G18','H18','J18','K18','L18','M18','N18','O18','P18','Q18','R18','S18','T18','XX',
    'XX', 'A17','B17','C17','D17','E17','F17','G17','H17','J17','K17','L17','M17','N17','O17','P17','Q17','R17','S17','T17','XX',
    'XX', 'A16','B16','C16','D16','E16','F16','G16','H16','J16','K16','L16','M16','N16','O16','P16','Q16','R16','S16','T16','XX',
    'XX', 'A15','B15','C15','D15','E15','F15','G15','H15','J15','K15','L15','M15','N15','O15','P15','Q15','R15','S15','T15','XX',
    'XX', 'A14','B14','C14','D14','E14','F14','G14','H14','J14','K14','L14','M14','N14','O14','P14','Q14','R14','S14','T14','XX',
    'XX', 'A13','B13','C13','D13','E13','F13','G13','H13','J13','K13','L13','M13','N13','O13','P13','Q13','R13','S13','T13','XX',
    'XX', 'A12','B12','C12','D12','E12','F12','G12','H12','J12','K12','L12','M12','N12','O12','P12','Q12','R12','S12','T12','XX',
    'XX', 'A11','B11','C11','D11','E11','F11','G11','H11','J11','K11','L11','M11','N11','O11','P11','Q11','R11','S11','T11','XX',
    'XX', 'A10','B10','C10','D10','E10','F10','G10','H10','J10','K10','L10','M10','N10','O10','P10','Q10','R10','S10','T10','XX',
    'XX', 'A9', 'B9', 'C9', 'D9', 'E9', 'F9', 'G9', 'H9', 'J9', 'K9', 'L9', 'M9', 'N9', 'O9', 'P9', 'Q9', 'R9', 'S9', 'T9', 'XX',
    'XX', 'A8', 'B8', 'C8', 'D8', 'E8', 'F8', 'G8', 'H8', 'J8', 'K8', 'L8', 'M8', 'N8', 'O8', 'P8', 'Q8', 'R8', 'S8', 'T8', 'XX',
    'XX', 'A7', 'B7', 'C7', 'D7', 'E7', 'F7', 'G7', 'H7', 'J7', 'K7', 'L7', 'M7', 'N7', 'O7', 'P7', 'Q7', 'R7', 'S7', 'T7', 'XX',
    'XX', 'A6', 'B6', 'C6', 'D6', 'E6', 'F6', 'G6', 'H6', 'J6', 'K6', 'L6', 'M6', 'N6', 'O6', 'P6', 'Q6', 'R6', 'S6', 'T6', 'XX',
    'XX', 'A5', 'B5', 'C5', 'D5', 'E5', 'F5', 'G5', 'H5', 'J5', 'K5', 'L5', 'M5', 'N5', 'O5', 'P5', 'Q5', 'R5', 'S5', 'T5', 'XX',
    'XX', 'A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4', 'H4', 'J4', 'K4', 'L4', 'M4', 'N4', 'O4', 'P4', 'Q4', 'R4', 'S4', 'T4', 'XX',
    'XX', 'A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3', 'J3', 'K3', 'L3', 'M3', 'N3', 'O3', 'P3', 'Q3', 'R3', 'S3', 'T3', 'XX',
    'XX', 'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'J2', 'K2', 'L2', 'M2', 'N2', 'O2', 'P2', 'Q2', 'R2', 'S2', 'T2', 'XX',
    'XX', 'A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'J1', 'K1', 'L1', 'M1', 'N1', 'O1', 'P1', 'Q1', 'R1', 'S1', 'T1', 'XX',
    'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX'
]

# boards lookup
BOARDS = {
     '9': board_9x9,
    '13': board_13x13,
    '19': board_19x19
}

# coords lookup
COORDS = {
     '9': coords_9x9,
    '13': coords_13x13,
    '19': coords_19x19,
}

# stones
EMPTY = 0
BLACK = 1
WHITE = 2
MARKER = 4
OFFBOARD = 7
LIBERTY = 8

# count
liberties = []
block = []

# current board used
board = None
coords = None

# GO ban size
BOARD_WIDTH = 0
BOARD_RANGE = 0
MARGIN = 2

# file markers
files = '     a b c d e f g h j k l m n o p q r s t'

# ASCII representation of stones
pieces = '.#o  bw +'

def print_board():
    # loop over board rows
    for row in range(BOARD_RANGE):
        # loop over board columns
        for col in range(BOARD_RANGE):
            # init square
            square = row * BOARD_RANGE + col

            # init stone
            stone = board[square]

            # print rank
            if col == 0 and row > 0 and row < BOARD_RANGE - 1:
                rank = BOARD_RANGE - 1 - row
                space = '  ' if len(board) == 121 else '   '
                print((space if rank < 10 else '  ') + str(rank), end='')

            # print board square's content
            print(pieces[stone] + ' ', end='')

        # print new line
        print()

    # print notation
    print(' ' + files[0:BOARD_RANGE*2] + '\n')

def print_fm(s, e):
    for i in range(s, e):
        print(f"board {i}")
        for j in range(0, 19):
            for k in range(0, 19):
                print(int(x[j][k][i]), end=" ")
            print()

def print_style_fm(style_fm, s, e):
    for i in range(s, e):
        print(f"board {i}")
        for j in range(0, 19):
            for k in range(0, 19):
                print(int(style_fm[j][k][i]), end=" ")
            print()


# set Go ban size
def set_board_size(command):
    # hook global variables
    global BOARD_WIDTH, BOARD_RANGE, board, coords

    # parse the board size
    size = int(command.split()[-1])

    # throw error if board size is not supported
    if size not in [19]:
        print('? current board size not supported\n')
        return

    # calculate current board size
    BOARD_WIDTH = size
    BOARD_RANGE = BOARD_WIDTH + MARGIN
    board = BOARDS[str(size)]
    coords = COORDS[str(size)]

# count liberties, save stone group coords
def count(square, color):
    # init piece
    piece = board[square]

    # skip offboard squares
    if piece == OFFBOARD: return

    # if there's a stone at square
    if piece and piece & color and (piece & MARKER) == 0:
        # save stone's coordinate
        block.append(square)

        # mark the stone
        board[square] |= MARKER

        # look for neighbours recursively
        count(square - BOARD_RANGE, color) # walk north
        count(square - 1, color)           # walk east
        count(square + BOARD_RANGE, color) # walk south
        count(square + 1, color)           # walk west

    # if the square is empty
    elif piece == EMPTY:
        # mark liberty
        board[square] |= LIBERTY

        # save liberty
        liberties.append(square)

# remove captured stones
def clear_block():
    for captured in block: board[captured] = EMPTY

# clear groups
def clear_groups():
    # hook global variables
    global block, liberties

    # clear block and liberties lists
    block = []
    liberties = []

# restore the board after counting stones
def restore_board():
    # clear groupd
    clear_groups()

    # unmark stones
    for square in range(BOARD_RANGE * BOARD_RANGE):
        # restore piece if the square is on board
        if board[square] != OFFBOARD: board[square] &= 3

# clear board
def clear_board():
    # clear groupd
    clear_groups()

    for square in range(len(board)):
        if board[square] != OFFBOARD: board[square] = 0
    global x
    x = np.zeros((19, 19, 19))  # feature map
    x[:, :, empty_board] = 1  # 空位盤面全設1
    # top5.sty_cnt = [[0, 0, 0], [0, 0, 0]]
    top5.update_queue.put((top5.new_board, ()))
    # print_fm(empty_board, empty_board + 1)

# make move on board
def set_stone(square, color):
    # make move on board
    board[square] = color

    # handle captures
    captures(3 - color)

# generate random move
def make_random_move(color):
    # find empty random square
    random_square = random.randrange(len(board))
    while board[random_square] != EMPTY:
        random_square = random.randrange(len(board))

    # make move
    set_stone(random_square, color)

    # count liberties
    count(random_square, color)

    # suicide move
    if len(liberties) == 0:
        # restore board
        restore_board()

        # take off the stone
        board[random_square] = EMPTY

        # search for another move
        try:
            # return non suicide move
            return make_random_move(color)
        except:
            # pass the move
            return ''

    # restore board
    restore_board()

    # return the move
    return coords[random_square]

# play command
def play(command):
    # parse color
    color = BLACK if command.split()[1] == 'B' else WHITE
    predict_next_move_str = predict_next_move(color)
    # parse square
    square_str = command.split()[-1]
    if square_str == 'pass':
        return
    col = ord(square_str[0]) - ord('A') + 1 - (1 if ord(square_str[0]) > ord('I') else 0)
    row_count = int(square_str[1:]) if len(square_str[1:]) > 1 else ord(square_str[1:]) - ord('0')
    row = (BOARD_RANGE - 1) - row_count
    square = row * BOARD_RANGE + col

    pos_x = row - 1
    pos_y = col - 1
    color_char = 'B' if color == BLACK else 'W'
    next_move = f'{color_char}[{numbertochar[pos_y]}{numbertochar[pos_x]}]'
    top5.sgf_string = top5.sgf_string + ';' + next_move + predict_next_move_str
    # show_pos(player_move, pos_x, pos_y, "player")

    now_board = 0 if color == BLACK else 1
    x[pos_x][pos_y][now_board] = 1  # 設定oppnent_board
    x[pos_x][pos_y][empty_board] = 0  # 設定empty_board
    set_last(pos_x, pos_y)  # 設定最後8步
    count_air(pos_x, pos_y)  # 計算氣數

    # make GUI move
    set_stone(square, color)
    # predict_style(pos_x, pos_y, color)

# handle captures
def captures(color):
    # loop over the board squares
    for square in range(len(board)):
        # init piece
        piece = board[square]

        # skip offboard squares
        if piece == OFFBOARD: continue

        # if stone belongs to the given color
        if piece & color:
            # count liberties
            count(square, color)

            # if no liberties left remove the stones
            if len(liberties) == 0: clear_block()

            # restore the board
            restore_board()

# 設定最後8步，pos_x和pos_y是最後一步的位置
def set_last(pos_x, pos_y):
    for i in range(last_8, last_1, -1):
        x[:, :, i] = x[:, :, i - 1]
    x[:, :, last_1] = 0
    x[pos_x, pos_y, last_1] = 1

# 計算氣數
liber = Queue()

def count_air(row, col):
    bfs = []
    # 找四周要bfs的
    now_color = -1
    if x[row][col][player_board] == 1:
        now_color = 1
    else:
        now_color = 0
    if col - 1 >= 0 and x[row][col - 1][empty_board] == 0:
        if x[row][col - 1][player_board] != now_color:
            bfs.insert(0, [row, col - 1])
        else:
            bfs.append([row, col - 1])
    if col + 1 < 19 and x[row][col + 1][empty_board] == 0:
        if x[row][col + 1][player_board] != now_color:
            bfs.insert(0, [row, col + 1])
        else:
            bfs.append([row, col + 1])
    if row - 1 >= 0 and x[row - 1][col][empty_board] == 0:
        if x[row - 1][col][player_board] != now_color:
            bfs.insert(0, [row - 1, col])
        else:
            bfs.append([row - 1, col])
    if row + 1 < 19 and x[row + 1][col][empty_board] == 0:
        if x[row + 1][col][player_board] != now_color:
            bfs.insert(0, [row + 1, col])
        else:
            bfs.append([row + 1, col])

    bfs.append([row, col])
    for m in bfs:
        BFS(m)
    bfs = []
    if not liber.empty():
        while not liber.empty():
            li_row, li_col = liber.get()
            if li_col - 1 >= 0 and x[li_row][li_col - 1][empty_board] == 0:
                bfs.append([li_row, li_col - 1])
            if li_col + 1 < 19 and x[li_row][li_col + 1][empty_board] == 0:
                bfs.append([li_row, li_col + 1])
            if li_row - 1 >= 0 and x[li_row - 1][li_col][empty_board] == 0:
                bfs.append([li_row - 1, li_col])
            if li_row + 1 < 19 and x[li_row + 1][li_col][empty_board] == 0:
                bfs.append([li_row + 1, li_col])
        for m in bfs:
            BFS(m)


def BFS(now):
    q = Queue()
    row = now[0]
    col = now[1]
    q.put((row, col))
    visited = [[False for _ in range(19)] for _ in range(19)]
    nodes = []
    air = 0
    next_board = player_air_1
    board = x[:, :, player_board]
    if x[row, col, oppnent_board] == 1:
        board = x[:, :, oppnent_board]
        next_board = oppnent_air_1
    while not q.empty():
        m, n = q.get()
        if m < 0 or n < 0 or m >= 19 or n >= 19 or visited[m][n]:
            continue
        visited[m][n] = True
        if board[m, n] == 1:
            nodes.append([m, n])
            q.put((m - 1, n))
            q.put((m + 1, n))
            q.put((m, n - 1))
            q.put((m, n + 1))
        elif x[m, n, empty_board] == 1:  # 空地 -> 自由度+1
            air += 1
    air = min(air, 4)
    for node in nodes:
        n_row = node[0]
        n_col = node[1]
        # 原本的改0，現在的氣數對應的棋盤改1
        for i in range(player_air_1, player_air_1 + 8):
            x[n_row][n_col][i] = 0
        if air == 0:
            x[n_row][n_col][empty_board] = 1
            x[n_row, n_col, player_board] = 0
            x[n_row, n_col, oppnent_board] = 0
            liber.put(node)
        else:
            x[n_row][n_col][next_board + air - 1] = 1

def top_5_preds_with_chars(predictions):
    tmps = [np.argpartition(prediction, -361)[-361:] for prediction in predictions]
    # print(tmps)
    resulting_preds_numbers = [
        np.flip(tmp[np.argsort(predictions[k][tmp])]) for k, tmp in enumerate(tmps)
    ]

    # print(f"predict:{predictions[0][resulting_preds_numbers[0]]}")
    return resulting_preds_numbers

# ai genarate move
def make_move(color):
    global x
    now_board = 0
    fm = x.copy()
    if color == WHITE:
        now_board = 1
        fm[:, :, 0] = x[:, :, oppnent_board]
        fm[:, :, 1] = x[:, :, player_board]
        fm[:, :, 2:6] = x[:, :, 6:10]
        fm[:, :, 6:10] = x[:, :, 2:6]

    feature_map = []
    feature_map.append(fm.copy())
    feature_map = np.array(feature_map)
    # print(np.shape(x))
    # print_board(0, 19)
    prediction = dan_model.predict(feature_map, verbose = 0)  # 預測
    next_chess = top_5_preds_with_chars(prediction)
    next_x, next_y, random_square = 0, 0, 0
    found_next_move = False
    #try to make a move and if doesn't work choose the second high possibility and so on.
    for i in range(361):
        next_x, next_y = next_chess[0][i] % 19, next_chess[0][i] // 19
        # print(f"next_x = {next_x}, next_y = {next_y}")
        if x[next_x][next_y][empty_board] == 1:
            # make move
            random_square = (next_x + 1) * 21 + next_y + 1
            set_stone(random_square, color)
            count(random_square, color)

            # suicide move
            if len(liberties) == 0:
                # restore board
                restore_board()

                # take off the stone
                board[random_square] = EMPTY
                continue

            # restore board
            restore_board()
            found_next_move = True
            break
    if not found_next_move:
        return 'PASS'

    # show_pos(model_move, next_x, next_y, "model")

    x[next_x][next_y][now_board] = 1  # 設定player_board
    x[next_x][next_y][empty_board] = 0  # 設定empty_board
    set_last(next_x, next_y)  # 設定最後8步
    count_air(next_x, next_y)  # 計算氣數

    # predict_style(next_x, next_y, color)
    color_char = 'B' if color == BLACK else 'W'
    next_move = f'{color_char}[{numbertochar[next_y]}{numbertochar[next_x]}]'
    top5.sgf_string = top5.sgf_string + ';' + next_move
    predict_next_move(WHITE if color == BLACK else BLACK)
    print(coords_19x19[(next_x + 1) * 21 + next_y + 1])
    return


# def predict_style(last_x, last_y, color):
#     model_style_fm = np.zeros((19, 19, 28))
#     if color == WHITE:
#         model_style_fm[:, :, 0] = x[:, :, 1]
#         model_style_fm[:, :, 1] = x[:, :, 0]
#         model_style_fm[:, :, 2:6] = x[:, :, 6:10]
#         model_style_fm[:, :, 6:10] = x[:, :, 2:6]
#     elif color == BLACK:
#         model_style_fm[:, :, 0:10] = x[:, :, 0:10]

#     model_style_fm[:, :, 10:19] = x[:, :, 10:19]

#     for rad in range(3, 0, -1):
#         row1 = max(0, last_x - rad)
#         row7 = min(18, last_x + rad)
#         col1 = max(0, last_y - rad)
#         col7 = min(18, last_y + rad)
#         for i in range(row1, row7 + 1, 1):
#             for j in range(col1, col7 + 1, 1):
#                 model_style_fm[i, j, round_7 + (3 - rad) * 3] = model_style_fm[i, j, 0]
#                 model_style_fm[i, j, round_7 + 1 + (3 - rad) * 3] = model_style_fm[i, j, 1]
#                 model_style_fm[i, j, round_7 + 2 + (3 - rad) * 3] = model_style_fm[i, j, empty_board]
#     fm = []
#     fm.append(model_style_fm.copy())
#     fm = np.array(fm)
#     prediction = style_model.predict(fm, verbose = 0)
#     prediction_number = np.argmax(prediction, axis=1)[0]

#     top5.update_queue.put((top5.show_style, (prediction_number, prediction[0][prediction_number] * 100, 'black' if color == BLACK else 'white')))
#     # top5.show_style(prediction_number, prediction[0][prediction_number] * 100, 'black' if color == BLACK else 'white')

def show_top_5_move():
    for i in range (20):
        print(f'top {i} move : {top_20_move[i][0]}, {top_20_move[i][1][0]:.2f}%')

def analyze_next_move(command):
    global x
    color = -1
    if command.split()[-1] == 'B':
        color = BLACK
    elif command.split()[-1] == 'W':
        color = WHITE
    # print(x)
    fm = np.zeros((19, 19, 19))
    if color == WHITE:
        fm[:, :, 0] = x[:, :, 1]
        fm[:, :, 1] = x[:, :, 0]
        fm[:, :, 2:6] = x[:, :, 6:10]
        fm[:, :, 6:10] = x[:, :, 2:6]
    elif color == BLACK:
        fm[:, :, 0:10] = x[:, :, 0:10]
    else:
        print("no color")
        return
    fm[:, :, empty_board:19] = x[:, :, empty_board:19]
    feature_map = []
    feature_map.append(fm)
    feature_map = np.array(feature_map)
    # print(np.shape(feature_map))
    prediction = dan_model.predict(feature_map, verbose = 0)  # 預測
    # print('in')
    next_chess = top_5_preds_with_chars(prediction)
    for i in range(5):
        next_x, next_y = next_chess[0][i] % 19, next_chess[0][i] // 19
        print(f'top {i} move : {coords_19x19[(next_x + 1) * 21 + next_y + 1]} {prediction[0][next_chess[0][i]]*100:.2f}')

def list_commands():
    print('Beta')

def predict_next_move(color):
    global x
    fm = x.copy()
    if color == WHITE:
        fm[:, :, 0] = x[:, :, oppnent_board]
        fm[:, :, 1] = x[:, :, player_board]
        fm[:, :, 2:6] = x[:, :, 6:10]
        fm[:, :, 6:10] = x[:, :, 2:6]

    feature_map = []
    feature_map.append(fm.copy())
    feature_map = np.array(feature_map)

    prediction = dan_model.predict(feature_map, verbose = 0)  # 預測
    next_chess = top_5_preds_with_chars(prediction)
    next_move_string = 'C['
    for i in range(20):
        top_x, top_y = next_chess[0][i] % 19, next_chess[0][i] // 19
        if i != 0:
            next_move_string = next_move_string + ', '
        next_move_string = next_move_string + f'Top{i + 1}:{numbertochar[top_y]}{numbertochar[top_x]}({prediction[0][next_chess[0][i]]:.4f})'
        top_20_move[i] = [coords_19x19[(top_x + 1) * 21 + top_y + 1], prediction[0][[next_chess[0][i]]]]
    top5.update_queue.put((top5.show_top5, (top_20_move, top_20_move)))
    next_move_string = next_move_string + ']'
    return next_move_string

def initial():
    global dan_model
    dan_model = load_model(FOLDER_PATH + '/model_Dan.h5')
    # 開始分析可以下的位置
    predict_next_move(BLACK)
    return

def solve_command():
    while True:
        # print('com')
        try:
            command = top5.command_queue.get(block=False)
            # print(command)
            if 'name' in command: print('= Go player imitation engine\n')
            elif 'protocol_version' in command: print('= 2\n')
            elif 'version' in command: print('=', VERSION, '\n')
            elif 'list_commands' in command: print('= ');list_commands();print('\n')
            elif 'boardsize' in command: set_board_size(command); print('=\n')
            elif 'clear_board' in command: print('= ');clear_board(); print('\n')
            elif 'initial' in command: print('= ');initial();print('\n')
            # elif '__select_opponent' in command: select_opponent()
            # elif 'select_opponent' in command: print('= ');select_opponent(); print('\n')
            # elif 'showfm' in command: print('= '); print_fm(0, 19); print('\n')
            # elif 'show_model_style_fm' in command: print('= '); print_style_fm(model_style_fm, 0, 28); print('\n')
            # elif 'show_player_style_fm' in command: print('= '); print_style_fm(player_style_fm, 0, 28); print('\n')
            elif 'showboard' in command: print('= '); print_board(); print('\n')
            elif 'show_top_5_move' in command: print('= '); show_top_5_move();print('\n')
            elif 'play' in command: play(command); print('=\n')
            elif 'genmove' in command: print('='); make_move(BLACK if command.split()[-1] == 'B' else WHITE);print('\n')
            elif 'quit' in command: print("=");print("\n");os._exit(0)
            elif 'analyze_next_move' in command: print('= '); analyze_next_move(command); print('\n')
            else: print('=\n') # skip currently unsupported commands
            if command != '__select_opponent':
                event.set()
                sys.stdout.flush()
        except queue.Empty:
            pass
        time.sleep(0.1)

# GTP communcation protocol
def gtp():
    global x
    x = np.zeros((19, 19, 19))
    x[:, :, empty_board] = 1
    # main GTP loop
    while True:
        # accept GUI command
        command = input()
        top5.command_queue.put(command)
        event.wait()
        # print('wait d')
        event.clear()
        # handle commands


# threading.Thread(target=Select_Difficulty.Select_Difficulty, daemon=True).start()
# start GTP communication
thread = None
top5_event = threading.Event()
event = threading.Event()
threading.Thread(target=solve_command, daemon=True).start()
threading.Thread(target=gtp, daemon=True).start()
top5.top5()
# while True:
#     top5_event.wait()
#     top5.top5()
#     top5_event.clear()

# gtp()
# select_opponent()
# time.sleep(3)
# select_opponent()

# threading.Thread(target=gtp, daemon=True).start()
# window.mainloop()
# select_opponent()
# time.sleep(3)
# select_opponent()

# set_board_size('boardsize 19')
# clear_board()
# analyze_next_move()
# play('play B D4')
# make_move(WHITE)
