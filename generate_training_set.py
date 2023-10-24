import os
import chess.pgn
from state import State

for fn in os.listdir("data"):
    pgn = open(os.path.join("data", fn))
    while 1:
        try:
            game = chess.pgn.read_game(pgn)
        except Exception:
            break
        value = {
            '1/2-1/2': 0,
            '0-1': -1,
            '1-0': 1
        }[game.headers["Result"]]
        board = game.board()
        for i, move in enumerate(game.mainline_moves()):
            board.push(move)
            # TODO: extract the boards
            print(value, State(board).serialize()[:,:,0])
    break