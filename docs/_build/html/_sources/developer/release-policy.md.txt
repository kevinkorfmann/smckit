# Release policy

Versions before 1.0 may correct beta APIs when migration notes and temporary
aliases are supplied. Version 1.0 freezes the public Python API, CLI, capability
registry, and result schema under semantic versioning.

A release candidate must pass clean wheel installation, the full supported
platform matrix, schema round trips, documentation, and all frozen oracle
specifications. A publication release additionally requires immutable benchmark
artifacts, an SBOM, container digests, archived inputs, and a Zenodo DOI.

Tags are signed where possible. PyPI publication uses trusted publishing from
the tag workflow. Upstream source pins and checksums are reviewed at every
release; changing an oracle requires a deliberate fixture and tolerance review.
