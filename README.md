# RL-Connect4

This repository includes several variants of reinforcement learning immplementations
using pyTorch to play the game of Connect4.

## PyTorch 101

First we gain some familiarity with [PyTorch](https://pytorch.org/) by
implementing a simple neural network to classify a sample dataset from
[scikit-learn](https://scikit-learn.org/stable/index.html).

In the [pytorch101](pytorch101) directory, there is `xgboost.py` and `nnet.py`.
The former uses the [XGBoost](https://xgboost.readthedocs.io/en/stable/#) library
to make a classifier. The latter uses PyTorch to implement a neural network.

```
> python xgb.py
> python nnet.py
```

![xgb](pytorch101/xgb.png)
![nnet](pytorch101/nnet.png)

## Visualization

We use python's [curses](https://docs.python.org/3/howto/curses.html) to make the
game playable in the terminal.

https://github.com/jovisly/RL-Connect4/assets/99686932/31859b85-a813-491c-9d30-e6e535a1ccf0

## Tabular Q-Learning

The "fun" thing about Connect 4 is that its state space is
3 ^ 42 = 109,418,989,131,512,359,209 (each grid can be occupied by player 1, 2,
or empty, and there are 42 grids). So we can't just train a Q lookup table.
Therefore we will use a neural network to approximate the Q table.

## DQN

The DQN (Deep Q-Network) algorithm uses a neural network to approximate the Q
table.
