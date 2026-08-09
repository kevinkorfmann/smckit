# SMC++ split promotion evidence

This directory retains the frozen public performance evidence for protocol
`sha256:c339dbb68e7ec26c721d909916edea5e388d77a60f03c04847e9daaa5cf560dd`.
The raw runner outputs are immutable and covered by `SHA256SUMS`.

`git-commit.txt` records the clean synthetic execution-snapshot commit used on
Sesame. `SOURCE_PROVENANCE.json` maps that narrowed snapshot to the canonical
repository commits and exact algorithm/worker hashes. The additional README and
source-provenance record were added after the run and are intentionally outside
the runner-generated checksum manifest.

The aggregate promotes only the frozen `split-control-v1` capability. It does
not establish a hardware-independent speed ratio or replace correctness oracles.
