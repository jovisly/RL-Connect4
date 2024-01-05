# RL-Connect4

This repository includes several variants of reinforcement learning immplementations
using pyTorch to play the game of Connect4.

## PyTorch 101

First we simply gain some familiarity with [PyTorch](https://pytorch.org/) by
implementing a simple neural network to classify a small dataset from
[scikit-learn](https://scikit-learn.org/stable/index.html).

In the [pytorch101](pytorch101) directory, there is `xgboost.py` and `nnet.py`.
The former uses the [XGBoost](https://xgboost.readthedocs.io/en/stable/#) library
to make a classifier. The latter uses PyTorch to implement a neural network.

```
> python xgb.py
> python nnet.py
```

![xgb](pytorch101/xgb.png)
