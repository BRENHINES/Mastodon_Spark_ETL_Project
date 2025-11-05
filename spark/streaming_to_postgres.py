# spark/streaming_to_postgres.py
import os, time, argparse, sys
import psycopg2
from mastodon import Mastodon, StreamListener, MastodonNetworkError

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--duration-sec", type=int, default=300, help="Durée max du stream (sec)")
parser.add_argument("--max-toots", type=int, default=500, help="Nb max de toots (0 = illimité)")
args = parser.parse_args()

# ---------- ENV ----------
MASTODON_INSTANCE = os.getenv("MASTODON_INSTANCE", "https://mastodon.social")
MASTODON_ACCESS_TOKEN = os.getenv("MASTODON_ACCESS_TOKEN")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "toots")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_TABLE = os.getenv("POSTGRES_TABLE", "toots")

REQUIRED = ["MASTODON_ACCESS_TOKEN", "POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]
missing = [k for k in REQUIRED if not os.getenv(k)]
if missing:
    print(f"❌ Missing env vars: {missing}", flush=True)
    sys.exit(2)

print("=" * 60)
print("🐘 Mastodon → PostgreSQL Writer (bounded run)")
print("=" * 60)
print(f"  Mastodon:   {MASTODON_INSTANCE}")
print(f"  PostgreSQL: {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
print(f"  Duration:   {args.duration_sec}s | Max toots: {args.max_toots or '∞'}")
print("=" * 60, flush=True)

# ---------- DB ----------
def connect_db(retries=5, wait=5):
    for i in range(retries):
        try:
            conn = psycopg2.connect(
                host=POSTGRES_HOST, port=POSTGRES_PORT,
                dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD
            )
            conn.autocommit = True
            print("✅ Connected to DB", flush=True)

            return conn
        except Exception as e:
            print(f"⏳ DB connect attempt {i+1}/{retries}: {e}", flush=True)
            time.sleep(wait)
    print("❌ Unable to connect DB", flush=True)
    sys.exit(3)

conn = connect_db()
cur = conn.cursor()
cur.execute(f"""
CREATE TABLE IF NOT EXISTS {POSTGRES_TABLE} (
    id TEXT PRIMARY KEY,
    ts TIMESTAMP,
    user_id TEXT,
    username TEXT,
    lang TEXT,
    content TEXT,
    hashtags TEXT[],
    favourites INT,
    reblogs INT
)
""")

# ---------- Client ----------
mastodon = Mastodon(api_base_url=MASTODON_INSTANCE, access_token=MASTODON_ACCESS_TOKEN, ratelimit_method='wait')

# ---------- Stats ----------
stats = {"inserted": 0, "errors": 0, "start": time.time()}

class Listener(StreamListener):
    def on_update(self, status):
        try:
            row = (
                str(status['id']),
                status['created_at'],
                str(status['account']['id']),
                status['account']['username'],
                status.get('language', 'unknown'),
                status['content'],
                [t['name'] for t in status.get('tags', [])],
                status.get('favourites_count', 0),
                status.get('reblogs_count', 0),
            )
            cur.execute(
                f"""INSERT INTO {POSTGRES_TABLE}
                    (id, ts, user_id, username, lang, content, hashtags, favourites, reblogs)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO NOTHING;""",
                row,
            )
            stats["inserted"] += 1

            print(f"✅ Inserted {stats['inserted']} | ID: {row[0]} User: {row[3]} Lang: {row[4]} Hashtags: {row[6]}", flush=True)
            if args.max_toots and stats["inserted"] >= args.max_toots:
                # Lever une exception custom pour casser la boucle extérieure proprement
                raise StopIteration
        except StopIteration:
            raise
        except Exception as e:
            stats["errors"] += 1
            print(f"❌ Insert error: {e}", flush=True)

listener = Listener()

# ---------- Run bounded ----------
end = time.time() + args.duration_sec
SLICE = 30  # secondes max par tranche
print("🚀 Streaming started (bounded)...", flush=True)

try:
    while time.time() < end:
        remaining = max(1, int(end - time.time()))
        timeout = min(SLICE, remaining)
        try:
            # stream_public bloque jusqu’à timeout ; lève souvent MastodonNetworkError “timed out”
            mastodon.stream_public(listener, timeout=timeout, run_async=False, reconnect_async=False)
        except StopIteration:
            break
        except MastodonNetworkError as e:
            # timeouts ou petits glitches réseau : on continue tant que la fenêtre n’est pas finie
            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                continue
            print(f"⚠️ Network warning: {e}", flush=True)
            time.sleep(1)
        except KeyboardInterrupt:
            print("🛑 Interrupted", flush=True)
            break
except Exception as e:
    print(f"❌ Fatal: {e}", flush=True)
    cur.close(); conn.close()
    sys.exit(1)

elapsed = time.time() - stats["start"]
rate = stats["inserted"] / elapsed if elapsed > 0 else 0.0

# ✅ AJOUTEZ CES LIGNES POUR DIAGNOSTIQUER
stop_reason = "unknown"
if args.max_toots and stats["inserted"] >= args.max_toots:
    stop_reason = f"max_toots_reached ({stats['inserted']}/{args.max_toots})"
elif time.time() >= end:
    stop_reason = f"timeout_reached ({elapsed:.1f}s/{args.duration_sec}s)"

print(f"✅ Done. Inserted={stats['inserted']} errors={stats['errors']} rate={rate:.2f}/s stop_reason={stop_reason}", flush=True)

cur.close()
conn.close()
sys.exit(0)
