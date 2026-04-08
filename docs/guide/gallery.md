# Gallery

This page is the visual check board for smckit. Each method gets two panels on
the same fixture or simulation: one from the native path and one from the
upstream or reference path. The x and y limits are matched inside each pair so
left/right comparison is direct rather than approximate.

The layout is intentionally gentle rather than dense. You should be able to
scan it and immediately see whether a method is behaving similarly across both
implementations.

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

   * - **Native**

       .. image:: ../gallery/esmc2_native.png
          :alt: eSMC2 native figure

       Native eSMC2 HMM quantities for a matched dormancy/selfing validation case.

     - **Upstream**

       .. image:: ../gallery/esmc2_upstream.png
          :alt: eSMC2 upstream figure

       Reference formulas translated directly from the original R implementation, on the same task and ranges.
```

## SMC++

```{eval-rst}
.. list-table::
   :widths: 1 1

   * - **Native**

       Native panel intentionally left blank for now.

       The current generated native SMC++ gallery output is not reliable enough
       to display here.

     - **Upstream**

       .. image:: ../gallery/smcpp_upstream.png
          :alt: SMC++ upstream figure

       A real computed run from the side Python environment on the same simulation.
```

## diCal2

```{eval-rst}
.. list-table::
   :widths: 1 1

   * - **Native**

       Native panel intentionally left blank for now.

       The current generated native diCal2 gallery output is not reliable
       enough to display here.

     - **Upstream**

       .. image:: ../gallery/dical2_upstream.png
          :alt: diCal2 upstream figure

       Matching ``diCal2.jar`` run on the same README example bundle.
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
