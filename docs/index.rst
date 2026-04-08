.. raw:: html

   <div class="smckit-hero">
    <h1>smckit: A home for SMC algorithms.</h1>
     <p>smckit keeps classic Sequentially Markovian Coalescent methods usable now while native Python implementations mature method by method.</p>
     <p>The core contract is simple: <code>implementation="upstream"</code> runs the original tool when ready, <code>implementation="native"</code> runs the in-repo port, and <code>implementation="auto"</code> prefers upstream fidelity. Use <code>smckit.upstream.status()</code> to see exactly which runtimes and bootstraps are ready on your machine.</p>
     <div class="smckit-pill-row">
       <span class="smckit-pill">Unified <code>SmcData</code> workflow</span>
       <span class="smckit-pill">Dual implementation API</span>
       <span class="smckit-pill">Public upstream runners</span>
       <span class="smckit-pill">Numba-native ports</span>
       <span class="smckit-pill">Vendored upstream references</span>
     </div>
   </div>

.. code-block:: python

   import smckit

   data = smckit.io.read_psmcfa("sample.psmcfa")
   data = smckit.tl.psmc(
       data,
       pattern="4+5*3+4",
       n_iterations=25,
       implementation="native",
   )
   smckit.pl.demographic_history(data)

.. raw:: html

   <div class="smckit-section-intro">
     Start with a quickstart if you want working code, use the method guide if you are choosing between tools, and only dive into internals if you are porting or validating an algorithm.
   </div>

Start Here
----------

.. raw:: html

   <div class="smckit-grid">
     <a class="smckit-card" href="get-started/introduction.html">
       <strong>Understand the philosophy</strong>
       <span>Why smckit exists, how upstream and native implementations coexist, and how to think about the project.</span>
     </a>
     <a class="smckit-card" href="guide/choosing-a-method.html">
       <strong>Choose a method</strong>
       <span>Pick the right SMC tool by sample type, inference target, and current implementation status.</span>
     </a>
     <a class="smckit-card" href="api/tl.html">
       <strong>See the tool API</strong>
       <span>Review the public inference functions and the shared <code>implementation=</code> selector.</span>
     </a>
     <a class="smckit-card" href="developer/parity.html">
       <strong>Inspect parity details</strong>
       <span>Read the fixture-by-fixture agreement notes behind the current status matrix.</span>
     </a>
   </div>

Implementation Status
---------------------

.. class:: smckit-section-intro

smckit treats implementation provenance as part of the public interface.
The matrix below is for users: it tells you which path exists today and whether
native parity is strong enough to be considered default-eligible in docs.
Runtime requirements still matter: `diCal2` needs a working Java runtime, `ASMC`
needs its C++/Boost build stack, and `SMC++` still depends on a controlled side
Python environment.

.. list-table::
   :header-rows: 1

   * - Method
     - Upstream
     - Native
     - Native default eligible
     - Tracked agreement
     - Notes
   * - PSMC
     - ✓
     - ✓
     - ✓
     - `0.9999223` lambda corr
     - Public upstream bridge now runs the vendored binary; native port remains stable.
   * - ASMC
     - ✓
     - ✓
     - ✓
     - `0.9984` MAP agreement
     - Public upstream path uses the documented ASMC executable outputs.
   * - MSMC2
     - ✓
     - ✓
     - ✓
     - `>= 0.999999865` lambda corr
     - Public upstream bridge runs the vendored MSMC2 CLI.
   * - MSMC-IM
     - ✓
     - ✓
     - ✗
     - `<= 3.36e-03` rel err on `N1/N2`
     - Public upstream bridge runs the vendored Python fitter; native fitter remains the tracked in-repo approximation.
   * - eSMC2
     - ✓
     - ✓
     - ✗
     - `-0.998936` Xi corr
     - Both paths exist; end-to-end native-vs-upstream fitting still needs work.
   * - SMC++
     - ✓
     - ✓
     - ✗
     - tracked native-vs-upstream log-`N_e` corr
     - Public upstream path exists, but the vendored source/bootstrap contract is still incomplete.
   * - diCal2
     - ✓
     - ✓
     - ✗
     - `2.27%` max rel loglik delta
     - Public upstream Java bridge parses the EM-path stdout into structured results; Java must be installed locally.
   * - SSM
     - ✗
     - ✓
     - ✗
     - —
     - Experimental extension framework rather than an upstream compatibility target.

.. class:: smckit-method-note

Detailed fixture notes, caveats, and remaining parity gaps live on
:doc:`developer/parity`.

Choose A Method At A Glance
---------------------------

.. class:: smckit-section-intro

Use this table when you already know the rough shape of your data or question.
The status matrix above answers what is runnable today; this guide answers
which method family to click into next.

.. class:: smckit-method-note

Starting from a VCF (or similar variant data) and metadata? Use this quick
decision path first, then use the table to pick the concrete method page.

Fast Decision Path (VCF + Metadata)
-----------------------------------

1. **Do you already have model files (`.param` / `.demo` / `.config`)?**

   - **Yes** -> :doc:`diCal2 <methods/dical2>`
   - **No** -> continue

2. **Are your genomes mostly unphased and do you have many samples?**

   - **Yes** -> :doc:`SMC++ <methods/smcpp>` (best fit for many unphased genomes)
   - **No** -> continue

3. **Do you have one diploid genome (or one pairwise sequence)?**

   - **Yes** -> :doc:`PSMC <methods/psmc>`
   - If dormancy/selfing is biologically important -> :doc:`eSMC2 <methods/esmc2>`
   - **No** -> continue

4. **Do you have phased haplotypes (typically 2-8 haplotypes)?**

   - **Yes** -> :doc:`MSMC2 <methods/msmc2>`
   - Need per-site pairwise TMRCA/local ancestry instead of one curve? -> :doc:`ASMC <methods/asmc>`
   - Need two-population migration curves after MSMC/MSMC2? -> :doc:`MSMC-IM <methods/msmc-im>`

Quick Feature-to-Tool Table
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 14 16 14 24 14

   * - Starting point
     - Phased?
     - Typical sample shape
     - Need `mu` / `r` upfront?
     - Best first tool
     - Next step
   * - VCF + demographic model files
     - optional
     - structured-model workflows
     - usually yes
     - :doc:`diCal2 <methods/dical2>`
     - tune model files and rerun fits
   * - VCF-like data, many genomes
     - mostly unphased
     - many individuals
     - yes
     - :doc:`SMC++ <methods/smcpp>`
     - compare native/upstream provenance
   * - single diploid genome
     - n/a
     - one genome
     - yes
     - :doc:`PSMC <methods/psmc>`
     - :doc:`eSMC2 <methods/esmc2>` for dormancy/selfing
   * - phased haplotypes
     - yes
     - 2-8 haplotypes
     - yes
     - :doc:`MSMC2 <methods/msmc2>`
     - :doc:`MSMC-IM <methods/msmc-im>` for migration summaries
   * - phased haplotypes + local ancestry target
     - yes
     - pairwise decoding
     - yes
     - :doc:`ASMC <methods/asmc>`
     - inspect per-site TMRCA tracks

.. list-table::
   :header-rows: 1
   :widths: 18 24 18 18 22

   * - Method
     - Best for
     - Input
     - Output focus
     - Current fit
   * - :doc:`PSMC <methods/psmc>`
     - one diploid genome
     - `.psmcfa`
     - `N_e(t)`
     - strong native baseline and the standard entry point for single-genome history inference
   * - :doc:`eSMC2 <methods/esmc2>`
     - one diploid genome with dormancy or selfing
     - `.psmcfa` or pairwise sequence input
     - `N_e(t)`, `beta`, `sigma`
     - native and upstream paths exist; safest when the R-backed upstream bridge is ready
   * - :doc:`MSMC2 <methods/msmc2>`
     - two to eight phased haplotypes
     - `.multihetsep`
     - coalescence rates and `N_e(t)`
     - strong fixed-fixture validation with an implementation surface that is still growing
   * - :doc:`MSMC-IM <methods/msmc-im>`
     - two-population MSMC or MSMC2 follow-up analysis
     - `.combined.msmc2.final.txt`
     - `N1(t)`, `N2(t)`, `m(t)`, `M(t)`
     - best when migration summaries matter more than raw coalescence-rate curves
   * - :doc:`SMC++ <methods/smcpp>`
     - many unphased genomes
     - `.smc` or `.smc.gz`
     - `N_e(t)`
     - upstream-backed runs are the safest choice today; the native solver is still tracked against them
   * - :doc:`ASMC <methods/asmc>`
     - per-site pairwise ancestry or TMRCA on phased haplotypes
     - hap/map/samples plus decoding quantities
     - TMRCA along the genome
     - strong native path for local ancestry-style questions rather than one population-size curve
   * - :doc:`diCal2 <methods/dical2>`
     - explicit structured demographic model files
     - `.param`, `.demo`, `.config`, sequences
     - sizes, growth, and migration parameters
     - best fit when the demographic specification already exists outside smckit
   * - :doc:`SSM Framework <methods/ssm>` / :doc:`PSMC-SSM <guide/psmc-ssm>`
     - method development, HMM inspection, and gradient fitting
     - `psmcfa`-style observations
     - differentiable PSMC and explicit state-space models
     - native-only experimental framework rather than an upstream compatibility target

Quickstarts
-----------

.. raw:: html

   <div class="smckit-grid">
     <a class="smckit-card" href="get-started/quickstart-psmc.html"><strong>PSMC</strong><span>One diploid genome, native port, classic entry point.</span></a>
     <a class="smckit-card" href="get-started/quickstart-asmc.html"><strong>ASMC</strong><span>Pairwise TMRCA decoding for phased haplotypes.</span></a>
     <a class="smckit-card" href="get-started/quickstart-msmc2.html"><strong>MSMC2</strong><span>Multiple haplotypes, validated fixtures, still growing.</span></a>
     <a class="smckit-card" href="get-started/quickstart-msmc-im.html"><strong>MSMC-IM</strong><span>Two-population migration fitting from MSMC output.</span></a>
     <a class="smckit-card" href="get-started/quickstart-esmc2.html"><strong>eSMC2</strong><span>Dormancy and selfing on top of pairwise SMC.</span></a>
     <a class="smckit-card" href="get-started/quickstart-smcpp.html"><strong>SMC++</strong><span>Many unphased genomes with upstream/native provenance.</span></a>
     <a class="smckit-card" href="get-started/quickstart-dical2.html"><strong>diCal2</strong><span>Structured demography from native model files.</span></a>
   </div>

.. toctree::
   :maxdepth: 2
   :caption: Get Started
   :hidden:

   get-started/introduction
   get-started/installation
   get-started/quickstart-psmc
   get-started/quickstart-asmc
   get-started/quickstart-msmc2
   get-started/quickstart-msmc-im
   get-started/quickstart-esmc2
   get-started/quickstart-smcpp
   get-started/quickstart-dical2

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   guide/smcdata
   guide/io-formats
   guide/choosing-a-method
   guide/interpreting-results
   guide/plotting
   guide/gallery
   guide/psmc-ssm

.. toctree::
   :maxdepth: 2
   :caption: Method Reference
   :hidden:

   methods/psmc
   methods/asmc
   methods/msmc2
   methods/msmc-im
   methods/esmc2
   methods/smcpp
   methods/dical2
   methods/ssm

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   :hidden:

   developer/contributing
   developer/architecture
   developer/adding-a-method
   developer/testing
   developer/parity
   developer/internals
   developer/internals-psmc
   developer/internals-smcpp
   developer/internals-esmc2
   developer/internals-msmc-im

.. toctree::
   :maxdepth: 2
   :caption: Agent Docs
   :hidden:

   agents/using-smckit
   agents/algorithms

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

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
