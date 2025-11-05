FROM bitnamilegacy/spark:3.5.1

USER root
# JAR Postgres dans le classpath Apache Spark
ADD https://jdbc.postgresql.org/download/postgresql-42.7.4.jar /opt/spark/jars/postgresql-42.7.4.jar

# log/opts optionnels (pas de /opt/bitnami/ ici)
RUN mkdir -p /opt/ivy/cache /opt/ivy/jars && chmod -R 777 /opt/ivy
COPY requirements.txt /opt/project/docker/spark/requirements.txt

RUN pip install --no-cache-dir -r /opt/project/docker/spark/requirements.txt
RUN pip install --no-cache-dir \
    "numpy>=1.21,<2.0" \
    "scipy>=1.7,<1.12" \
    "pandas>=1.3,<2.0" \
    "scikit-learn>=1.0,<1.3" \
    psycopg2-binary

ENV SPARK_IVY=/opt/ivy
USER root
