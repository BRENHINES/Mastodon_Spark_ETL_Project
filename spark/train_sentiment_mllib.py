import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, length
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Configuration
CSV = os.getenv("SENTIMENT140_CSV", "/opt/project/dagster-project/data/sentiment140.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MODEL_NAME = os.getenv("MODEL_NAME", "sentiment_lr_v1")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# PostgreSQL config (not used for training, but kept for consistency)
PG_HOST = os.getenv('POSTGRES_HOST', 'postgres')
PG_PORT = os.getenv('POSTGRES_PORT', '5432')
PG_DB = os.getenv('POSTGRES_DB', 'toots')
PG_USER = os.getenv('POSTGRES_USER', 'postgres')
PG_PASS = os.getenv('POSTGRES_PASSWORD', 'postgres')


def log(msg):
    """Print with flush for real-time logging"""
    print(msg, flush=True)


def main():
    log("\n" + "=" * 60)
    log("🤖 SENTIMENT ANALYSIS TRAINING")
    log("=" * 60)

    # Check CSV exists
    if not Path(CSV).exists():
        log(f"❌ ERROR: CSV file not found: {CSV}")
        return 1

    file_size_mb = Path(CSV).stat().st_size / (1024 * 1024)
    log(f"✅ CSV found: {CSV}")
    log(f"   Size: {file_size_mb:.2f} MB")

    # Create Spark session
    log("\n🔧 Creating Spark session...")
    spark = (SparkSession.builder
             .master("local[*]")
             .appName("train-sentiment")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.driver.memory", "1g")
             .config("spark.executor.memory", "1g")
             .getOrCreate())

    spark.sparkContext.setLogLevel("WARN")
    log("✅ Spark session created")

    try:
        # Load data
        log("\n📊 Loading Sentiment140 dataset...")
        df_raw = (spark.read
                  .option("header", "false")
                  .option("mode", "DROPMALFORMED")
                  .option("encoding", "utf-8")
                  .csv(CSV))

        log("✅ CSV loaded")

        # Map columns (Sentiment140 format: target, id, date, query, user, text)
        log("\n🔧 Mapping columns...")
        df = df_raw.select(
            col("_c0").cast("int").alias("label_raw"),
            col("_c5").cast("string").alias("text")
        )

        # Remove nulls
        df = df.dropna(subset=["label_raw", "text"])

        # Remove empty texts
        df = df.filter(length(col("text")) > 0)

        total_count = df.count()
        log(f"✅ Loaded {total_count:,} rows")

        if total_count == 0:
            log("❌ ERROR: No data after filtering!")
            return 1

        # Convert label: 0 (negative) and 4 (positive) -> 0 and 1
        log("\n🔧 Converting labels...")
        df = df.withColumn(
            "label",
            when(col("label_raw") == 4, lit(1.0)).otherwise(lit(0.0))
        )

        # Show distribution
        log("\n📊 Sentiment Distribution:")
        df.groupBy("label").count().orderBy("label").show()

        # Sample if too large (for faster training)
        if total_count > 100000:
            log(f"\n⚠️  Dataset large ({total_count:,} rows)")
            log("   Sampling 100K rows for faster training...")
            df = df.sample(fraction=100000 / total_count, seed=42)
            sampled_count = df.count()
            log(f"✅ Sampled: {sampled_count:,} rows")

        # Cache for performance
        df = df.cache()
        log("💾 DataFrame cached")

        # Build ML Pipeline
        log("\n🔧 Building ML pipeline...")
        log("   - RegexTokenizer")
        log("   - StopWordsRemover")
        log("   - HashingTF (10K features)")  # ✅ Reduced from 262K!
        log("   - IDF")
        log("   - LogisticRegression")

        tok = RegexTokenizer(
            inputCol="text",
            outputCol="tokens",
            pattern="\\W+"
        )

        sw = StopWordsRemover(
            inputCol="tokens",
            outputCol="clean"
        )

        # ✅ FIXED: Reduced from 1<<18 (262K) to 10K features
        tf = HashingTF(
            inputCol="clean",
            outputCol="tf",
            numFeatures=10000  # Much more reasonable!
        )

        idf = IDF(
            inputCol="tf",
            outputCol="features"
        )

        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            regParam=0.01,  # ✅ Added regularization
            maxIter=20  # ✅ Reduced from 50
        )

        pipe = Pipeline(stages=[tok, sw, tf, idf, lr])
        log("✅ Pipeline built")

        # Split train/test
        log("\n✂️  Splitting train/test (80/20)...")
        train, test = df.randomSplit([0.8, 0.2], seed=42)

        train_count = train.count()
        test_count = test.count()
        log(f"✅ Train: {train_count:,} rows")
        log(f"✅ Test:  {test_count:,} rows")

        # Train model
        log("\n🚀 Training model...")
        log("   (This may take 1-3 minutes...)")
        model = pipe.fit(train)
        log("✅ Training complete!")

        # Evaluate
        log("\n📈 Evaluating model...")
        pred = model.transform(test)

        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )

        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )

        acc = evaluator_acc.evaluate(pred)
        f1 = evaluator_f1.evaluate(pred)

        log(f"✅ Accuracy: {acc:.4f}")
        log(f"✅ F1 Score: {f1:.4f}")

        # Show sample predictions
        log("\n📋 Sample Predictions:")
        pred.select("text", "label", "prediction").show(5, truncate=50)

        # Save model
        log(f"\n💾 Saving model to: {MODEL_PATH}")
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        model.write().overwrite().save(MODEL_PATH)
        log("✅ Model saved successfully!")

        # ✅ REMOVED: PostgreSQL registry write (not needed for this project)
        # You can add it later if needed, but for now it's causing the error

        log("\n" + "=" * 60)
        log("✅ SUCCESS!")
        log(f"   Model Path: {MODEL_PATH}")
        log(f"   Accuracy:   {acc:.4f}")
        log(f"   F1 Score:   {f1:.4f}")
        log(f"   Train Size: {train_count:,}")
        log(f"   Test Size:  {test_count:,}")
        log("=" * 60)

        return 0

    except Exception as e:
        log(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        log("\n🛑 Stopping Spark...")
        spark.stop()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
