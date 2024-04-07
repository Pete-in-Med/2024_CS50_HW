"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    num_X = sum (row.count(X) for row in board)
    num_O = sum (row.count(O) for row in board)

    if num_X == num_O:
        return X
    else:
        return O



def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i,j))
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    current_player = player(board)
    #深層複製 (應該是把整個矩陣切成很多個部分，像現在的狀況就是把board用row的方式切成一列一列的，存成一個新的矩陣)
    new_board = [row[:] for row in board]
    new_board[i][j] = current_player
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows
    #第一種寫法
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not None:
            return board [i][0]
    #第二種寫法
    #for row in board:
        #(這邊的變數設立我也沒有到很清楚，不知道為什麼row[0]會是一個元素而不是一整列)
        #if row.count(row[0]) == 3 and row[0] is not None:
            #return row[0]

    # Check columns
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] and board[0][j] is not None:
            return board[0][j]

    # Check diagonals
    #第一種寫法
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
        return board[0][2]
    return None
    #第二種寫法
    #if board[0][0] == board[1][1] == board [2][2] and board [0][0] is not None:
        #return board[0][0]
    #elif board[2][0] == board[1][1] == board [0][2] and board[2][0] is not None:
        #return board [2][0]
    #else:
        #return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    if sum(row.count(EMPTY) for row in board) == 0:
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    def max_value(board):
        if terminal(board):
            return utility(board)
        v = float("-inf")
        for action in actions(board):
            #當做了action，會導致board往result的方向前進，而min_value就去了解min這個選手，會選擇怎麼樣的最優解，最後再選出最高也就是最適合我的值)
            v = max(v, min_value(result(board, action)))
        return v

    def min_value(board):
        if terminal(board):
            return utility(board)
        v = float("inf")
        for action in actions(board):
            v = min(v, max_value(result(board, action)))
        return v

    current_player = player(board)
    if current_player == X:
        #這邊我沒有到全盤的了解，但大體上就是，有很多的actions，其中我選擇其中一個action來看
        #在這個action中，board會往新的result跑，而所有的result一訂有個對O來講的最優解，用min_value表示
        #此比較中有不一樣類型的數據，因此採用lambda作為比較的參考，最後把所有最佳解排序，找到最適合自己的，也就是max_value)
        return max(actions(board), key=lambda action: min_value(result(board, action)))
    else:
        return min(actions(board), key=lambda action: max_value(result(board, action)))
