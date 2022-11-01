FROM python:3.9-slim as base

ENV PYTHONFAULTHANDLER=1 \
	PYTHONUNBUFFERED=1 \
	PYTHONHASHSEED=random

WORKDIR /home

FROM base as builder

ENV PIP_NO_CACHE_DIR=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	PIP_DEFAULT_TIMEOUT=100 \
	POETRY_VERSION=1.2.0

RUN pip install "poetry==$POETRY_VERSION"
RUN python -m venv /venv

COPY pyproject.toml poetry.lock ./
RUN . /venv/bin/activate && poetry install --without dev --no-root

COPY src/ README.md ./
RUN . /venv/bin/activate && poetry build

FROM base as final

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install --no-install-recommends libx11-dev
RUN apt-get -y install --no-install-recommends libgl-dev
RUN apt-get -y install --no-install-recommends libxrender-dev

COPY --from=builder /venv /venv
COPY --from=builder /home/dist .
RUN . /venv/bin/activate && pip install *.whl

ENV PATH="/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/venv"

RUN useradd -ms /bin/bash nonroot
RUN chown nonroot /mnt/
USER nonroot
