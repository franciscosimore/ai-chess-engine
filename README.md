# An AI Chess Engine

Simple one-look-ahead neural network value function.

## Usage

./play.py runs the webserver on localhost:5000

## Implementation

The board is serialized intro a 8x8x5 bitvector.

Trained net can be found in ./nets/value_100k.pth. It takes in a serialized board and outputs a range from -1 (chances of black winning) to 1 (chances of white winning).

## Training set

The value function was trained on 100k chess moves (examples) from http://caissabase.co.uk/