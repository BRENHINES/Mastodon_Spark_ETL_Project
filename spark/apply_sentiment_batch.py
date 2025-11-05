import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, element_at
from pyspark.ml.pipeline import PipelineModel

PG_HOST = os.getenv("POSTGRES_HOST", "postgres")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_DB   = os.getenv("POSTGRES_DB", "toots")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")
PG_URL  = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"

MODEL_DIR  = os.getenv("MODEL_DIR", "/models")
MODEL_NAME = os.getenv("MODEL_NAME", "sentiment_lr_v1")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

spark = (SparkSession.builder
         .appName("ml-apply-sentiment")
         .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS","4"))
         .getOrCreate())

# Charger le modèle
model = PipelineModel.load(MODEL_PATH)

# Charger toots (on score surtout en anglais, car modèle entraîné sur EN)
toots = (spark.read.format("jdbc")
         .option("url", PG_URL)
         .option("dbtable", "toots")
         .option("user", PG_USER)
         .option("password", PG_PASS)
         .option("driver", "org.postgresql.Driver")
         .load()
        ).select("id","ts","user_id","username","lang","content")

toots_en = toots.filter(col("content").isNotNull() & (col("lang").isNull() | (col("lang") == "en")))

scored = model.transform(toots_en)

# probas: vecteur [p_negative, p_positive] pour LogisticRegression
# Avec element_at(proba, 1/2) car vecteur MLlib est indexé à partir de 1 côté SQL expr
from pyspark.sql.functions import col as c
prob_pos = element_at(c("probability"), 2).alias("prob_pos")
prob_neg = element_at(c("probability"), 1).alias("prob_neg")

out = (scored
       .select(
           "id","ts","user_id","username","lang","content",
           col("prediction").cast("int").alias("pred"),
           prob_pos, prob_neg
       )
       .withColumn("sentiment_label", when(col("pred")==1, lit("positive")).otherwise(lit("negative")))
       .withColumn("sentiment_score", col("prob_pos"))
      )

# Option: éviter de re-scorrer ce qui existe déjà
# On récupère les id déjà scorés si table existe
from pyspark.sql.utils import AnalysisException
existing_ids = None
try:
    existing_ids = (spark.read.format("jdbc")
                    .option("url", PG_URL)
                    .option("dbtable", "(select id from toots_with_sentiment) as t")
                    .option("user", PG_USER)
                    .option("password", PG_PASS)
                    .option("driver", "org.postgresql.Driver")
                    .load())
except AnalysisException:
    pass

if existing_ids is not None:
    out = out.join(existing_ids, on="id", how="left_anti")

# Écriture
(out.write.format("jdbc")
 .option("url", PG_URL)
 .option("dbtable", "toots_with_sentiment")
 .option("user", PG_USER)
 .option("password", PG_PASS)
 .option("driver", "org.postgresql.Driver")
 .mode("append")
 .save())

print("✓ Sentiment scoring written to table toots_with_sentiment")
spark.stop()
