# PSMC+ preservation image

This OCI definition runs the immutable PSMC+ source pin with the dependency
versions reported by upstream. Build from the repository root so the pinned
`vendor/PSMCplus` submodule is the only source copied into the image. This
preservation image deliberately targets Linux x86-64 because the upstream-pinned
2022 SciPy/Numba stack does not supply a complete Linux ARM64 wheel set:

```bash
git submodule update --init vendor/PSMCplus
docker build -f preservation/containers/psmcplus/Dockerfile \
  -t smckit-psmcplus:032168f .
```

The base image is content-addressed. Python dependencies are version- and
hash-pinned in `requirements.lock`. The image defaults every supported numeric
runtime to one thread; callers may explicitly override those variables.

Run the frozen constant-population smoke fixture:

```bash
docker run --rm -v "$PWD:/work" smckit-psmcplus:032168f \
  -in /opt/PSMCplus/simulations/constpopsize.mhs \
  -D 4 -b 100 -its 1 -thresh 0 -c 1 -o /work/oracle_
```

Docker can export the image as an OCI archive, and Apptainer can consume either
that archive or an image published to an OCI registry:

```bash
docker save smckit-psmcplus:032168f -o psmcplus-032168f.tar
apptainer build psmcplus-032168f.sif docker-archive://psmcplus-032168f.tar
```

This image preserves the original `PSMCplus.py` entry point. The separate
`simulate_HMM.py` entry point remains available by overriding the entry point:

```bash
docker run --rm --entrypoint python smckit-psmcplus:032168f \
  /opt/PSMCplus/simulate_HMM.py --help
```
