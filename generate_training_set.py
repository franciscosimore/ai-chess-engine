import os
import chess.pgn
from state import State
import numpy as np

def get_dataset(num_samples=None):
    X, Y = [], []
    values = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}
    gn = 0
    for fn in os.listdir("data"):
        pgn = open(os.path.join("data", fn))
        while 1:
            try:
                game = chess.pgn.read_game(pgn)
            except Exception:
                break
            print(f"parsing game {gn}, got {len(X)} examples")
            result = game.headers["Result"]
            if result not in values:
                continue
            value = values[result]
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                ser = State(board).serialize()
                X.append(ser)
                Y.append(value)
            if num_samples is not None and len(X) > num_samples:
                return X, Y
            gn += 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

if __name__ == "__main__":
    X, Y = get_dataset(1e6)
    np.savez("data/processed/caissabase_1m.npz", X, Y)