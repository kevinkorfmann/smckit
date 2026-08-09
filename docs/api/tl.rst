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

.. autofunction:: smcpp_cross_validate

diCal2
------

.. autofunction:: dical2

PHLASH
------

PHLASH is a maintained external Python integration. ``implementation="auto"``
and ``implementation="upstream"`` execute the installed PHLASH package;
``implementation="native"`` is intentionally unsupported.

.. autofunction:: phlash

PSMC+
-----

PSMC+ provides a typed, normalized adapter to the immutable upstream
implementation and an independent native fit/decode engine. Explicit native
execution is parity-enforced; ``auto`` remains conservative pending the broader
empirical validation matrix.

.. autoclass:: PSMCPlusOptions
   :members:

.. autofunction:: psmcplus
