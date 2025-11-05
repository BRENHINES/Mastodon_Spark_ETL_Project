import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

CSV = os.getenv("SENTIMENT140_CSV", "/opt/project/dagster-project/data/sentiment140.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MODEL_NAME = "sentiment_lr_en"

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

PG_HOST = os.getenv('POSTGRES_HOST','postgres')
PG_PORT = os.getenv('POSTGRES_PORT','5432')
PG_DB   = os.getenv('POSTGRES_DB','toots')
PG_USER = os.getenv('POSTGRES_USER','postgres')
PG_PASS = os.getenv('POSTGRES_PASSWORD','postgres')

PG_URL  = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"

spark = (SparkSession.builder
         .appName("train-sentiment")
         .config("spark.sql.shuffle.partitions","2")
         .getOrCreate())

# Sentiment140: colonnes classiques (label, id, date, query, user, text) — adapter si ton CSV diffère
df_raw = (spark.read
          .option("header", "false")
          .option("multiLine","false")
          .csv(CSV))

# Mappe les colonnes
df = df_raw.select(
    col("_c0").cast("int").alias("label_raw"),
    col("_c5").cast("string").alias("text")
).dropna(subset=["label_raw","text"])

# Label binaire: 0 = negatif (label_raw=0), 1 = positif (label_raw=4)
df = df.withColumn("label", when(col("label_raw") == 4, lit(1.0)).otherwise(lit(0.0)))

# Pipeline: tokenize -> remove stopwords -> TF -> IDF -> LR
tok  = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W+")
sw   = StopWordsRemover(inputCol="tokens", outputCol="clean")
tf   = HashingTF(inputCol="clean", outputCol="tf", numFeatures=1<<18)
idf  = IDF(inputCol="tf", outputCol="features")
lr   = LogisticRegression(featuresCol="features", labelCol="label", regParam=0.0, maxIter=50)

pipe = Pipeline(stages=[tok, sw, tf, idf, lr])

train, test = df.randomSplit([0.8, 0.2], seed=42)
model = pipe.fit(train)

pred = model.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
acc = evaluator.evaluate(pred)
f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(pred)

# Sauvegarde modèle
model.write().overwrite().save(MODEL_PATH)

# Register dans Postgres
metrics = spark.createDataFrame([(
    "sentiment_lr_en", "v1", acc, f1, train.count()+test.count()
)], schema="model_name string, model_version string, accuracy double, f1 double, training_rows long")

(metrics.selectExpr(
    f"'{MODEL_NAME}' as model_name",
    "'v1' as model_version",
    "current_timestamp() as trained_at",
    "training_rows",
    "accuracy",
    "f1",
    "cast(null as string) as notes"
).write
 .format("jdbc")
 .option("url", PG_URL)
 .option("dbtable", "ml_model_registry")
 .option("user", PG_USER)
 .option("password", PG_PASS)
 .option("driver", "org.postgresql.Driver")
 .mode("append")
 .save())

print(f"✓ Model saved to {MODEL_PATH} | accuracy={acc:.3f} f1={f1:.3f}")
spark.stop()
