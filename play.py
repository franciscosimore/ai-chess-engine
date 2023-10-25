import chess
import torch
from state import State
from train import Net

class Valuator(object):
    def __init__(self):
        vals = torch.load("nets/value_100k.pth", map_location=lambda storage, loc: storage)
        self.model = Net()
        self.model.load_state_dict(vals)
    
    def __call__(self, s):
        brd = s.serialize()[None]
        output = self.model(torch.tensor(brd).float())
        return float(output.data[0][0])

def explore_leaves(s, v):
    ret = []
    for e in s.edges():
        s.board.push(e)
        ret.append((v(s), e))
        s.board.pop()
    return ret

if __name__ == "__main__":
    v = Valuator()
    s = State()
    l = explore_leaves(s, v)
    print(l)
    # while not s.board.is_game_over():
    #     l = explore_leaves(s, v)
    #     print(l)