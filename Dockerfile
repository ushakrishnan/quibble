# UNTESTED: written for CUDA 12.8 / torch 2.7.0 and a non-root user with UID/GID 1000.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ushak && useradd -m -u ${UID} -g ${GID} ushak
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Requirements include all model backends and may replace the base image's Torch build.
COPY --chown=ushak:ushak . .
USER ushak
ENV QUIBBLE_DATA_HOME=/app/.cache/openml
ENTRYPOINT ["python", "-m"]
CMD ["quibble.doctor"]
