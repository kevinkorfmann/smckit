smckit
======

A unified Python framework for Sequentially Markovian Coalescent (SMC)
demographic inference methods.

smckit reimplements PSMC, MSMC2, SMC++, and future SMC methods under a
single API inspired by `scanpy <https://scanpy.readthedocs.io>`_. All
numerical kernels are JIT-compiled with Numba, matching or exceeding the
speed of the original C implementations.

.. code-block:: python

   import smckit

   data = smckit.io.read_psmcfa("sample.psmcfa")
   data = smckit.tl.psmc(data, pattern="4+5*3+4", n_iterations=25)
   smckit.pl.demographic_history(data)

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   gallery
   api/index

Benchmarks
----------

On NA12878 chr22 (10 EM iterations):

.. list-table::
   :header-rows: 1

   * - Backend
     - E-step
     - Full run
     - vs C
   * - **Numba JIT**
     - 0.52s
     - **5.4s**
     - **0.85x (17% faster)**
   * - C PSMC (reference)
     - --
     - 6.3s
     - 1.0x
   * - NumPy (pure Python)
     - 6.1s
     - ~65s
     - 10.2x
