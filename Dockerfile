# Container execution remains unverified; optional models are explicit build extras.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.11 /uv /uvx /bin/
ARG UID=1000
ARG GID=1000
ARG DEPENDENCY_EXTRAS="web,data"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && apt-get clean
RUN groupadd -g ${GID} quibble && useradd -m -u ${UID} -g ${GID} quibble
WORKDIR /app
COPY --chown=quibble:quibble . .
RUN set -eu; set --; for extra in $(printf '%s' "$DEPENDENCY_EXTRAS" | tr ',' ' '); do \
        set -- "$@" --extra "$extra"; \
    done; uv sync --locked --no-dev "$@"; chown ${UID}:${GID} /app
USER quibble
ENV PATH="/app/.venv/bin:$PATH"
ENV QUIBBLE_DATA_HOME=/app/.cache/openml
ENTRYPOINT ["python", "-m"]
CMD ["quibble_ml.doctor"]
