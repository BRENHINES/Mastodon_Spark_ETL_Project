import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, to_date, length, avg, count, lower, explode, split, trim
)

# --- Config Postgres (recycle celles de ton .env) ---
PG_HOST = os.getenv("POSTGRES_HOST", "postgres")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_DB   = os.getenv("POSTGRES_DB", "toots")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")
PG_URL  = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"

spark = (SparkSession.builder
         .appName("batch-from-postgres")
         .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS","4"))
         .getOrCreate())

# 1) Lire la table 'toots'
src = (spark.read.format("jdbc")
       .option("url", PG_URL)
       .option("dbtable", "toots")
       .option("user", PG_USER)
       .option("password", PG_PASS)
       .option("driver", "org.postgresql.Driver")
       .load())

# 2) Normalisation minimale (ts en timestamp, longueur contenu)
base = (src
        .withColumn("ts_ts", to_timestamp(col("ts")))
        .filter(col("ts_ts").isNotNull() & col("content").isNotNull())
        .withColumn("day", to_date(col("ts_ts")))
        .withColumn("toot_len", length(col("content")))
        .select("id","ts_ts","day","user_id","username","lang","content","hashtags","toot_len")
       ).cache()

# 3) Agrégations

# 3.1 daily_counts
daily_counts = (base
                .groupBy("day")
                .agg(count("*").alias("toot_count"),
                     avg("toot_len").alias("avg_len")))

# 3.2 top_hashtags (si 'hashtags' tableau est rempli par le stream, on l’explose)
#     sinon on retombe sur une extraction naïve depuis 'content'.
if "hashtags" in base.columns:
    tags_df = base.select("day", explode(col("hashtags")).alias("hashtag"))
else:
    tokens = base.select("day", split(lower(col("content")), r"\s+").alias("tokens"))
    tags_df = (tokens
               .select("day", explode(col("tokens")).alias("tok"))
               .filter(col("tok").rlike(r"^#\w+"))
               .withColumn("hashtag", trim(col("tok")))
               .select("day", "hashtag"))

top_hashtags = (tags_df
                .groupBy("day","hashtag")
                .agg(count("*").alias("occurrences")))

# 3.3 user_activity
user_activity = (base
                 .groupBy("day","user_id","username")
                 .agg(count("*").alias("toot_count"),
                      avg("toot_len").alias("avg_len")))

# 4) Write helpers
def write_df(df, table, mode="overwrite"):
    (df.write.format("jdbc")
     .option("url", PG_URL)
     .option("dbtable", table)
     .option("user", PG_USER)
     .option("password", PG_PASS)
     .option("driver", "org.postgresql.Driver")
     .mode(mode)
     .save())

# 5) Écriture (V1: overwrite complet)
write_df(daily_counts, "daily_counts", "overwrite")
write_df(top_hashtags, "top_hashtags", "overwrite")
write_df(user_activity, "user_activity", "overwrite")

print("✓ Batch terminé.")
spark.stop()
