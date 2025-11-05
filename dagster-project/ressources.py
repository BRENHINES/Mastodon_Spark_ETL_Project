import os
import subprocess
from dagster import ConfigurableResource

class SparkSubmitResource(ConfigurableResource):
    submit_bin: str = "/opt/bitnami/spark/bin/spark-submit"
    master: str = "spark://spark-master:7077"
    extra_args: str = ""

    def submit(self, app_py: str, app_args: list[str] | None = None, env: dict | None = None):
        cmd = [self.submit_bin, "--master", self.master]
        if self.extra_args:
            cmd += self.extra_args.split()
        cmd += [app_py]
        if app_args:
            cmd += app_args
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, env={**os.environ, **(env or {})}, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        return result

class PgEnvResource(ConfigurableResource):
    host: str
    port: int = 5432
    db: str
    user: str
    password: str

    @property
    def sqlalchemy_url(self):
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"
