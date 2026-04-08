IO: Input / Output
==================

.. currentmodule:: smckit.io

Functions for reading and writing the native input/output formats of every
SMC tool that smckit wraps. See :doc:`/guide/io-formats` for a conceptual
overview of each format.

PSMC
----

.. autofunction:: read_psmcfa
.. autofunction:: read_psmc_output
.. autofunction:: write_psmc_output

MSMC / MSMC2
------------

.. autofunction:: read_multihetsep
.. autofunction:: read_msmc_output
.. autofunction:: read_msmc_combined_output

MSMC-IM
-------

.. autofunction:: read_msmc_im_output
.. autofunction:: write_msmc_im_output

ASMC
----

.. autofunction:: read_asmc
.. autofunction:: read_decoding_quantities
.. autofunction:: read_hap
.. autofunction:: read_samples
.. autofunction:: read_map

SMC++
-----

.. autofunction:: read_smcpp_input

diCal2
------

.. autofunction:: read_dical2
.. autofunction:: read_dical2_param
.. autofunction:: read_dical2_demo
.. autofunction:: read_dical2_rates
.. autofunction:: read_dical2_config
.. autofunction:: read_dical2_sequences
