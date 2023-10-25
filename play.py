import chess
import chess.svg
from flask import Flask, Response
import torch
from state import State
from train import Net
import time

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

v = Valuator()
s = State()

def computer_move():
    move = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)[0]
    print(move)
    s.board.push(move[1])

app = Flask(__name__)

@app.route("/")
def hello():
    return """
<html>
  <head>
    <script src='https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js'></script>
    <style>
      button {
        font-size: 20px;
      }
    </style>
  </head>
  <body>
    <img width="600" height="600" src="board.svg?%f"/><br>
    <button onclick="console.log('Button Clicked'); $.post('/move', function(data) { console.log('Response:', data); location.reload(); });">Make Computer Move</button>
  </body>
</html>
""" % time.time()

@app.route("/board.svg")
def board():
    return Response(chess.svg.board(board=s.board), mimetype="image/svg+xml")

@app.route("/move", methods=["POST"])
def move():
    computer_move()
    return("")

if __name__ == "__main__":
    app.run(debug=True)

# Self Play
# while not s.board.is_game_over():
#     l = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
#     move = l[0]
#     print(move)
#     s.board.push(move[1])
# print(s.board.result())