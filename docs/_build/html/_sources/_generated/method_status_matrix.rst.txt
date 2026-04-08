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
     - `0.9999223 lambda corr`
     - Public upstream bridge now runs the vendored binary; native port remains stable.
   * - ASMC
     - ✓
     - ✓
     - ✓
     - `0.9984 MAP agreement`
     - Public upstream path uses the documented ASMC executable outputs.
   * - MSMC2
     - ✓
     - ✓
     - ✓
     - `>= 0.999999865 lambda corr`
     - Public upstream bridge runs the vendored MSMC2 CLI.
   * - MSMC-IM
     - ✓
     - ✓
     - ✓
     - `strict payload match on 4-case upstream-backed oracle matrix`
     - Public upstream bridge runs the vendored Python fitter; native and upstream now share the same public payload contract and are interchangeable on the enforced Yoruba/French oracle matrix.
   * - eSMC2
     - ✓
     - ✓
     - ✗
     - `<= 0.3605% max tracked Xi rel err`
     - Both paths exist; tracked fit parity is enforced for fixed-rho, rho-redo, beta, sigma, beta+sigma, and grouped pop_vect=[3,3], with upstream still preferred outside that matrix.
   * - SMC++
     - ✓
     - ✓
     - ✓
     - `>= 0.999125 tracked one-pop log-Ne corr`
     - Public upstream path exists; the native one-pop path now clears the strict small control and larger tracked `.smc` parity fixtures, with fixed-model E-step statistics matching upstream on the same matrix.
   * - diCal2
     - ✓
     - ✓
     - ✗
     - `README exp fixed-point dloglik=7.45e-4; explicit starts <=2.21e-4; full search dloglik=9.73e-2`
     - Public upstream Java bridge parses the EM-path stdout into structured results; docs now show both README exp panels, but the current full native search still misses the upstream fit and IM parity is still open.
   * - SSM
     - n/a
     - ✓
     - ✗
     - `—`
     - Novel in-repo extension framework rather than an upstream compatibility target.
