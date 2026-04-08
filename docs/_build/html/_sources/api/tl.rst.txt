Tools: Inference Algorithms
===========================

.. currentmodule:: smckit.tl

The inference tools. Each function takes an :class:`~smckit.SmcData`
container, runs an SMC method, and writes results into ``data.results[<name>]``.
See :doc:`/guide/choosing-a-method` for help picking the right one.

All public ``smckit.tl`` algorithms follow the same implementation-provenance
contract:

- ``implementation="native"`` runs the in-repo implementation
- ``implementation="upstream"`` requests the original tool when a bridge exists
- ``implementation="auto"`` resolves to the best available path and currently
  prefers upstream when upstream is exposed

PSMC
----

.. autofunction:: psmc

eSMC2
-----

.. autofunction:: esmc2

MSMC2
-----

.. warning::
   MSMC2 is in development and not yet fully validated. See
   :doc:`/methods/msmc2`.

.. autofunction:: msmc2

MSMC-IM
-------

.. autofunction:: msmc_im

ASMC
----

.. autofunction:: asmc

SMC++
-----

.. autofunction:: smcpp

diCal2
------

.. autofunction:: dical2
