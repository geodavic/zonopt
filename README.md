# zonopt 

![Tests](https://github.com/geodavic/zonopt/actions/workflows/test.yml/badge.svg?event=push)

A `pytorch` implementation of the results from [On Finding the Closest Zonotope to a Polytope in Hausdorff Distance](https://web.ma.utexas.edu/users/gdavtor/)

## Installation

Using poetry:
```bash
poetry install
```

Using pip:
```
pip install .
```

## Quickstart

To use `zonopt`, you will need a polytope `P` that you want to approximate by a zonotope. Instantiate `P` by specifying its vertices:

```python
>>> from zonopt import Polytope
>>> P = Polytope(points=[[0,0],[1,0],[0,1],[1,1]])
```

To perform optimization, implement a `ZonotopeTrainer`. This requires an `Optimizer`, which specifies the learning rate and learning rate schedule, and a starting zonotope rank.

```python
>>> from zonopt.train import ZonotopeTrainer, Optimizer
>>> optim = Optimizer(lr=0.001)
>>> trainer = ZonotopeTrainer(target_polytope=P, optimizer=optim, zonotope_rank=4)
```

The `zonotope_rank` specifies the rank of the zonotopes you are searching. In this case, it will start from a random zonotope with that given rank. You can also pass your own starting zonotope `Z` to the trainer using `start=Z` instead of passing the rank (see below how to initialize a zonotope). 

To train, call the `train()` method with the desired number of steps:

```python
>>> trainer.train(500)
```

This will run subgradient descent for 500 steps and return the resulting zonotope.


## Using `zonopt`
