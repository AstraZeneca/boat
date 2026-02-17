
ARG BASE_IMAGE=harbor.csis.astrazeneca.net/azimuth-iron/cai-bp/cpu:latest
ARG RUNTIME_IMAGE=harbor.csis.astrazeneca.net/python/python:3.10-slim-bookworm

FROM ${BASE_IMAGE} AS builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

COPY ./src ./src

# Replace the artifactory repository based on your project.

RUN --mount=type=secret,id=artifactory_token \
    TOKEN="$(cat /run/secrets/artifactory_token)" && \
    poetry config repositories.az-artifactory https://artifactory.astrazeneca.com/artifactory/api/pypi/pypi-virtual && \
    poetry config http-basic.az-artifactory cai-eng@astrazeneca.com ${TOKEN} && \
    poetry install

FROM ${RUNTIME_IMAGE} AS runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY ./src ./src

ENTRYPOINT ["python", "./src/main.py"]

CMD ["--help"]
