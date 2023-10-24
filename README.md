# An AI Chess Engine from scratch

We can think of chess decisions as a big search tree.
Then use a neural network to prune the search tree.

Pieces (1+6*2 = 13):
* Blank.
* Pawn.
* Bishop.
* Knight.
* Rook.
* Queen.
* King.

Definition: value network.
V = f(state)

state(board):

Extra states:
* Castle available x2
* En passant available x2
8x8x4+4 = 260 bits (vector of 0 or 1)