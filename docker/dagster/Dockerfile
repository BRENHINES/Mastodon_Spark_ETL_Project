FROM python:3.11-slim
RUN apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*
ENV DAGSTER_HOME=/opt/dagster
RUN useradd -m -u 1000 dagster && mkdir -p ${DAGSTER_HOME}/logs && chown -R dagster:dagster ${DAGSTER_HOME}
WORKDIR /opt/project
COPY requirements.txt /opt/project/docker/dagster/requirements.txt
RUN pip install --no-cache-dir -r /opt/project/docker/dagster/requirements.txt
USER dagster
