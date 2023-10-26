import chess
import chess.svg
import traceback
from flask import Flask, Response, request
import torch
from state import State
from train import Net
import time
import base64

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
    board_svg = base64.b64encode(chess.svg.board(board=s.board).encode("utf-8")).decode("utf-8")
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <script src='https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js'></script>
        <style>
            body {
                text-align: center;
                font-family: Arial, sans-serif;
            }

            #chessboard {
                width: 700px;
                height: 600px;
                margin: 0 auto;
                border: 2px solid #333;
                background-color: #eee;
            }

            #controls {
                margin-top: 20px;
            }

            input[type="text"] {
                padding: 5px;
                margin-right: 10px;
            }

            button {
                padding: 10px 20px;
                background-color: #007BFF;
                color: #fff;
                border: none;
                cursor: pointer;
            }

            button:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <h1>Chess Game</h1>
        <div id="chessboard">
            <img width="600" height="600" src="data:image/svg+xml;base64,%s"</img>
        </div>
        <div id="controls">
            <button id="computerMoveButton" onclick='console.log("Button Clicked"); $.post("/computer", function(data) { console.log("Response:", data); location.reload(); });'>Computer Move</button></br></br>
            <form id="humanMoveForm" action="/human">
                <label for="chessMove">Enter Your Chess Move:</label>
                <input type="text" id="chessMove" name="move" placeholder="e.g., e2e4" required>
                <input type="submit" id="humanMoveButton" value="Human Move">
            </form>
        </div>
    </body>
    </html>
""" % board_svg

@app.route("/computer", methods=["POST"])
def get_computer_move():
    computer_move()
    return hello()

@app.route("/human", methods=["GET"])
def get_human_move(automatic_response=False):
    if not s.board.is_game_over():
        move = request.args.get("move", default="")
        if move is not None and move != "":
            print("Human moves", move)
            try:
                s.board.push_san(move)
            except Exception:
                traceback.print_exc()
            if automatic_response == True:
              computer_move()
    else:
        print("Game Over :(")
    return hello()

# TODO: self-play
# @app.route("/selfplay", methods=["GET"])
# def selfplay():
#   s = State()
#   ret 
#   while not s.board.is_game_over():
#       l = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
#       move = l[0]
#       print(move)
#       s.board.push(move[1])
#   print(s.board.result())

if __name__ == "__main__":
    app.run(debug=True)