FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 verireason

WORKDIR /workspace
COPY . /workspace

RUN python -m pip install --no-cache-dir \
      --requirement requirements/m10-cpu-lock.txt \
    && python -m pip install --no-cache-dir --no-build-isolation --no-deps .

RUN chown -R verireason:verireason /workspace
USER verireason

RUN make m10-verify PYTHON=python PIP_CHECK_PYTHON=python
CMD ["make", "m10-verify", "PYTHON=python", "PIP_CHECK_PYTHON=python"]
