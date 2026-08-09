FROM --platform=linux/amd64 python:3.10.5-slim-bullseye@sha256:f9f03f46267e182193544299504687e711c623e2a085323138f94ed9b01ce641

ENV MPLBACKEND=Agg \
    NUMBA_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY preservation/containers/psmcplus/requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock

COPY vendor/PSMCplus /opt/PSMCplus
WORKDIR /work

ENTRYPOINT ["python", "/opt/PSMCplus/PSMCplus.py"]
