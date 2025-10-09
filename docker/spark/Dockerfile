FROM bitnami/spark:3.5
USER root
ADD https://jdbc.postgresql.org/download/postgresql-42.7.4.jar /opt/spark/jars/postgresql-42.7.4.jar
COPY spark-defaults.conf /opt/bitnami/spark/conf/spark-defaults.conf
USER 1001
