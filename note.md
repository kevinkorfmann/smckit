## Gallery TODO

- ensure msmc2 panel actually re-runs the upstream D implementation rather than relying solely on fixture arrays; ideally record the command and store new outputs (and mark whether the D binary is runnable in CI).
- document the data sources for each gallery pair (PSMC, MSMC-IM, SMC++, diCal2, eSMC2) so we know which ones are generated live versus using reference fixtures/formulas; include any environment requirements (Rscript path, side Python env) so future renders stay reproducible.
- confirm that the new matched axis strategy stays aligned with the automated figure generation script; add a regression check that the generated PNGs are newer than their source runs when CI rebuilds the docs.
