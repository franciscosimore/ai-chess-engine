# An AI Chess Engine

We can think of chess decisions as a big search tree.
Then use a neural network to prune the search tree.

Definition: value network.
V = f(state)
-> -1: black wins board state
-> 0: draw board state
-> 1: white wins board state

**state(board):**
Pieces (2+7*2 = 16):
* Universal
** Blank
** Blank (en passant)
* Pieces
** Pawn
** Bishop
** Knight
** Rook
** Rook (can castle)
** Queen
** King
Extra states:
* To move

8x8x5 = 320 bits (vector of 0 or 1)