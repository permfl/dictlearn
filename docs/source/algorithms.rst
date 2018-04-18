.. _algorithms:

=====================
High-Level Algorithms
=====================

High level interfaces to the different algorithms and methods. Using algorithms
from `optimize` we get denoising, inpainting etc

Training
********

.. autoclass:: dictlearn.algorithms.Trainer
    :members:

.. autoclass:: dictlearn.algorithms.ImageTrainer
    :members:
    :inherited-members:


Denoise
*******
.. autoclass:: dictlearn.algorithms.Denoise
    :members:
    :inherited-members:




Inpaint
*******

.. autoclass:: dictlearn.algorithms.Inpaint
    :members:
    :inherited-members:

.. autoclass:: dictlearn.algorithms.TextureSynthesis
    :members:
    :inherited-members:


Structure Detection
*******************

.. autofunction:: dictlearn.detection.smallest_cluster
