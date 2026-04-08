# Gallery

This page is the visual check board for smckit. Most methods get two panels on
the same fixture or simulation: one from the native path and one from the
upstream or reference path. When a raw left/right pair would overstate current
agreement, the gallery switches to scoped validation figures instead. The x
and y limits are matched inside each pair so comparison is direct rather than
approximate.

The layout is intentionally gentle rather than dense. You should be able to
scan it and immediately see whether a method is behaving similarly across both
implementations, and where the comparison is only claimed on a tracked oracle
fixture matrix rather than across arbitrary runs.

## PSMC

```{eval-rst}
.. list-table::
   :widths: 1 1

   * - **Native**

       .. image:: ../gallery/psmc_native.png
          :alt: PSMC native figure

       smckit on the bundled ``NA12878_chr22.psmcfa`` example.

     - **Upstream**

       .. image:: ../gallery/psmc_upstream.png
          :alt: PSMC upstream figure

       The reference ``.psmc`` output shown on the same scale.
```

## ASMC

```{eval-rst}
.. list-table::
   :widths: 1 1

   * - **Native**

       .. image:: ../gallery/asmc_native.png
          :alt: ASMC native figure

       smckit posterior means and MAP state calls on the vendored ``n300`` array example.

     - **Upstream**

       .. image:: ../gallery/asmc_upstream.png
          :alt: ASMC upstream figure

       The matching reference decoding outputs on the same genomic span and y-ranges.
```

## MSMC2

```{eval-rst}
.. list-table::
   :widths: 1 1

   * - **Native**

       .. image:: ../gallery/msmc2_native.png
          :alt: MSMC2 native figure

       smckit on ``data/msmc2_test.multihetsep`` with the known fixture overlaid.

     - **Upstream**

       .. image:: ../gallery/msmc2_upstream.png
          :alt: MSMC2 upstream figure

       The preserved fixture curve on the same axes.
```

## MSMC-IM

```{eval-rst}
.. list-table::
   :widths: 1 1

   * - **Native**

       .. image:: ../gallery/msmc_im_native.png
          :alt: MSMC-IM native figure

       The fitted ``N1``, ``N2``, and migration history from smckit on the Yoruba-French example.

     - **Upstream**

       .. image:: ../gallery/msmc_im_upstream.png
          :alt: MSMC-IM upstream figure

       The official ``MSMC_IM.py`` output on the exact same task and axes.
```

## eSMC2

```{eval-rst}
.. list-table::
   :widths: 1 1

   * - **Tracked Demography**

       .. image:: ../gallery/esmc2_parity_demography.png
          :alt: eSMC2 tracked parity demography figure

       Oracle-backed native/upstream overlays on the clean 800 bp fixture for
       the six enforced fit branches: fixed rho, rho redo, beta, sigma,
       beta+sigma, and grouped ``pop_vect=[3,3]``.

     - **Transition Agreement**

       .. image:: ../gallery/esmc2_parity_transition.png
          :alt: eSMC2 tracked parity transition figure

       Final-state transition matrices for representative challenging branches.
       This is a scoped parity check on tracked fixtures, not a blanket claim
       for every eSMC2 input shape.
```

## SMC++

```{eval-rst}
.. list-table::
   :widths: 1 1

   * - **Native**

       .. image:: ../gallery/smcpp_native.png
          :alt: SMC++ native figure

       The tracked native one-pop path on the bundled larger `.smc`
       parity fixture.
       The panel includes the same shape and scale diagnostics enforced by the
       tracked one-pop regression matrix.

     - **Upstream**

       .. image:: ../gallery/smcpp_upstream.png
          :alt: SMC++ upstream figure

       A real computed run from the side Python environment on the same
       tracked fixture and axis ranges.
```

## diCal2

```{eval-rst}
.. list-table::
   :widths: 1 1

   * - **Native**

       .. image:: ../gallery/dical2_native.png
          :alt: diCal2 native figure

       Current independent native run on the README `exp` bundle.
       Oracle-point and explicit-start parity on this fixture are now close to
       upstream, but the current full meta-start search still misses the
       upstream best fit.

     - **Upstream**

       .. image:: ../gallery/dical2_upstream.png
          :alt: diCal2 upstream figure

       Matching `diCal2.jar` run on the same README `exp` bundle.
```

## SSM

```{eval-rst}
.. list-table::
   :widths: 1 1

   * - **Native**

       .. image:: ../gallery/ssm_native.png
          :alt: SSM native figure

       The current PSMC-backed SSM path on ``NA12878 chr22``.

     - **Reference**

       .. image:: ../gallery/ssm_reference.png
          :alt: SSM reference status figure

       The matching ``smckit.tl.psmc()`` baseline on the same task and axis ranges.
```
