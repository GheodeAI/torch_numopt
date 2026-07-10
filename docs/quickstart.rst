Quickstart
==========

This guide will walk you through the basic usage of `torch_numopt`. You will learn how to:

- Set up an objective function for a supervised learning problem.
- Choose and configure an optimizer.
- Run a training loop.
- Understand the closure-based workflow.

For a complete list of available optimizers and advanced customization, see the :doc:`/api_reference`.

Installation
------------

Install the package via pip:

.. code-block:: bash

   pip install torch-numopt

Or install from source:

.. code-block:: bash

   git clone https://github.com/GheodeAI/torch_numopt.git
   cd torch_numopt
   pip install -e .


1. Define your model and loss
-----------------------------

Start with a standard PyTorch model and loss function:

.. code-block:: python

   import torch
   import torch.nn as nn

   model = nn.Sequential(
       nn.Linear(10, 20),
       nn.ReLU(),
       nn.Linear(20, 1)
   )
   loss_fn = nn.MSELoss()


2. Prepare your data
--------------------

Generate some dummy data (or load your own):

.. code-block:: python

   X = torch.randn(100, 10)
   y = torch.randn(100, 1)


3. Create the objective
-----------------------

The :class:`~torch_numopt.objective.SupervisedLearningObjective` wraps your model, loss function, and data. It also holds a reference to the optimizer.

.. code-block:: python

   from torch_numopt import SupervisedLearningObjective, GaussNewtonLS

   optimizer = GaussNewtonLS(model.parameters(), lr_init=1.0)
   objective = SupervisedLearningObjective(model, loss_fn, optimizer)

   # Important: set the data before the first step
   objective.set_data(X, y)

**Why do we call `set_data`?** The objective is **stateless** with respect to the data – this lets you switch between full-batch and mini-batches at each iteration by calling `set_data` again before `optimizer.step()`.


4. Run the training loop
------------------------

The optimizer expects a closure that computes the loss and performs backpropagation. The objective itself is callable and does exactly that, so you simply pass it to `optimizer.step()`.

.. code-block:: python

   for epoch in range(100):
       # If you want to use a different batch, call objective.set_data(batch_X, batch_y) here
       optimizer.step(objective)   # objective.closure() is called internally

       # Monitor loss (optional)
       with torch.no_grad():
           loss = objective.loss(*objective.params)
           print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

**Important:** This is **not** the typical PyTorch pattern of calling `loss.backward()` and then `optimizer.step()`. The optimizer takes full control of evaluation and backpropagation, which is necessary for line-search and trust-region methods that re-evaluate the objective multiple times per iteration.


5. Try a different optimizer
----------------------------

Switching to another optimizer is trivial. For example, to use Newton with line search:

.. code-block:: python

   from torch_numopt import NewtonLS

   optimizer = NewtonLS(
       model.parameters(),
       lr_init=1.0,
       damping="identity",      # improves stability
       mu=1e-4,
       block_hessian=True       # saves memory
   )

   objective = SupervisedLearningObjective(model, loss_fn, optimizer)
   objective.set_data(X, y)

   # Training loop remains the same
   for epoch in range(100):
       optimizer.step(objective)
       # ...


6. Using mini-batches
---------------------

If you want to use mini-batch training, simply call `set_data` with a new batch before each `step()`:

.. code-block:: python

   batch_size = 32
   for epoch in range(100):
       for i in range(0, len(X), batch_size):
           batch_X = X[i:i+batch_size]
           batch_y = y[i:i+batch_size]
           objective.set_data(batch_X, batch_y)
           optimizer.step(objective)

**Caveat:** The library is designed for deterministic (full-batch) problems. Mini-batch updates introduce noise that can destabilize second-order methods. If you experience issues, consider using a larger batch size or switching to a first-order optimizer (e.g., `GradientDescentLS`).


7. Where to go next
-------------------

- Browse the :doc:`available algorithms <api_reference.available_algorithms>` for all ready-to-use optimizers.
- Learn how to :doc:`build <api_reference.building_algorithms>` your own custom optimizer.
- Read the :doc:`module documentation <auto/torch_numopt>` for detailed API of all classes and functions.