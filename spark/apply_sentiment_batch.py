import os, sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lit
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

MODEL_PATH = "file:///models/sentiment_lr_v1"
PG_HOST = os.getenv('POSTGRES_HOST', 'postgres')
PG_PORT = os.getenv('POSTGRES_PORT', '5432')
PG_DB   = os.getenv('POSTGRES_DB', 'toots')
PG_USER = os.getenv('POSTGRES_USER', 'postgres')
PG_PASS = os.getenv('POSTGRES_PASSWORD', 'postgres')
JDBC_URL = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"
SOURCE_TABLE = "toots"
TARGET_TABLE = "toots_with_sentiment"

def log(x): print(x, flush=True)

def main():
    log("\n" + "="*60)
    log("🎯 APPLYING SENTIMENT MODEL TO TOOTS")
    log("="*60)
    log(f"\n📋 Configuration:\n   Model Path:    {MODEL_PATH}\n   PostgreSQL:    {PG_HOST}:{PG_PORT}/{PG_DB}\n   Source Table:  {SOURCE_TABLE}\n   Target Table:  {TARGET_TABLE}")

    # Spark local
    spark = (SparkSession.builder
             .master("local[*]")
             .appName("apply-sentiment")
             .config("spark.jars", "/opt/spark/jars/postgresql-42.7.4.jar")
             .config("spark.sql.shuffle.partitions","4")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    try:
        # Load model
        log(f"\n📦 Loading model from: {MODEL_PATH}")
        model = PipelineModel.load(MODEL_PATH)
        log("✅ Model loaded successfully!")

        # Read toots
        log(f"\n📊 Loading toots from table: {SOURCE_TABLE}")
        df = (spark.read.format("jdbc")
              .option("url", JDBC_URL)
              .option("dbtable", SOURCE_TABLE)
              .option("user", PG_USER)
              .option("password", PG_PASS)
              .option("driver", "org.postgresql.Driver")
              .load()
             ).select("id","ts","user_id","username","lang","content")

        df = df.filter(col("content").isNotNull() & (col("content") != ""))
        n = df.count()
        log(f"✅ Loaded {n:,} toots")
        if n == 0:
            log("⚠️  No toots to score. Exiting cleanly.")
            return 0

        log("\n📋 Sample toots:")
        df.select("id","content","ts").show(3, truncate=50)

        # Prepare + score
        log("\n🔧 Preparing data for model...")
        df_for_model = df.withColumnRenamed("content", "text")

        log("\n🚀 Applying sentiment model...")
        predictions = model.transform(df_for_model)
        log("✅ Predictions generated!")

        # Extract probas from Vector -> array
        probs = vector_to_array("probability")
        final_df = (predictions
            .select(
                "id","ts","user_id","username","lang","text",
                probs.getItem(1).alias("prob_pos"),   # P(positive)
                probs.getItem(0).alias("prob_neg"),   # P(negative)
                F.col("prediction").cast("int").alias("pred")
            )
            .withColumnRenamed("text", "content")
            .withColumn("sentiment_label", when(col("pred")==1, lit("positive")).otherwise(lit("negative")))
            .withColumn("sentiment_score", col("prob_pos"))
        )

        log("\n📋 Sample predictions:")
        final_df.select("content","sentiment_label","sentiment_score").show(5, truncate=50)

        log("\n📊 Sentiment Distribution:")
        final_df.groupBy("sentiment_label").count().orderBy("sentiment_label").show()

        # Write to Postgres (append pour rejouer sans tout écraser)
        log(f"\n💾 Saving to PostgreSQL table: {TARGET_TABLE}")
        (final_df.write.format("jdbc")
         .option("url", JDBC_URL)
         .option("dbtable", TARGET_TABLE)
         .option("user", PG_USER)
         .option("password", PG_PASS)
         .option("driver", "org.postgresql.Driver")
         .mode("overwrite")  # ou "append" si tu veux conserver l'historique
         .save())
        log(f"✅ Saved {final_df.count():,} rows to {TARGET_TABLE}")

        log("\n" + "="*60 + "\n✅ SUCCESS!\n" + "="*60)
        return 0

    except Exception as e:
        log(f"\n❌ FATAL ERROR: {e}")
        import traceback; traceback.print_exc()
        return 1
    finally:
        log("\n🛑 Stopping Spark..."); spark.stop(); log("✅ Done")

if __name__ == "__main__":
    sys.exit(main())
