#!/usr/bin/env python3
"""
Environment & Infrastructure Health Check Script.

Connects to PostgreSQL, Redis, and MLflow services using environment variables
loaded from .env (or environment defaults), prints PASS/FAIL per service, and
exits non-zero on any connection failure.
"""

import os
import sys
import socket
import urllib.request
import urllib.error

# Load environment variables from .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try importing database / client libraries
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


def check_postgres():
    """Verify connection to PostgreSQL database."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    db = os.getenv("POSTGRES_DB", "supply_chain")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    
    if HAS_PSYCOPG2:
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                dbname=db,
                user=user,
                password=password,
                connect_timeout=3
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            cursor.fetchone()
            conn.close()
            return True, f"Connected to PostgreSQL database '{db}' on {host}:{port}"
        except Exception as e:
            return False, f"PostgreSQL connection failed on {host}:{port}/{db} -> {e}"
    else:
        try:
            s = socket.create_connection((host, port), timeout=3)
            s.close()
            return True, f"TCP connection established to PostgreSQL on {host}:{port}"
        except Exception as e:
            return False, f"PostgreSQL socket test failed on {host}:{port} -> {e}"


def check_redis():
    """Verify connection to Redis cache service."""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    
    if HAS_REDIS:
        try:
            r = redis.Redis(host=host, port=port, db=db, socket_timeout=3)
            if r.ping():
                return True, f"Received PONG from Redis on {host}:{port}"
            else:
                return False, f"Redis ping returned unexpected response on {host}:{port}"
        except Exception as e:
            return False, f"Redis connection failed on {host}:{port} -> {e}"
    else:
        try:
            s = socket.create_connection((host, port), timeout=3)
            s.close()
            return True, f"TCP connection established to Redis on {host}:{port}"
        except Exception as e:
            return False, f"Redis socket test failed on {host}:{port} -> {e}"


def check_mlflow():
    """Verify connection to MLflow Tracking Server."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000").rstrip('/')
    health_url = f"{uri}/health"
    root_url = f"{uri}/"
    
    for url in [health_url, root_url]:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'HealthCheck/1.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status in (200, 302, 301):
                    return True, f"MLflow Tracking Server responsive at {uri} (HTTP {resp.status})"
        except urllib.error.HTTPError as e:
            if e.code in (200, 302, 301, 404):
                return True, f"MLflow Tracking Server responded at {uri} (HTTP {e.code})"
        except Exception:
            continue

    return False, f"Could not reach MLflow Tracking Server at {uri}"


def main():
    print("=" * 60, flush=True)
    print("      Supply Chain Platform Infrastructure Check      ", flush=True)
    print("=" * 60, flush=True)
    
    results = {
        "PostgreSQL": check_postgres(),
        "Redis": check_redis(),
        "MLflow": check_mlflow()
    }
    
    all_passed = True
    for service, (passed, msg) in results.items():
        status_tag = "PASS" if passed else "FAIL"
        print(f"[{status_tag}] {service:<12} : {msg}", flush=True)
        if not passed:
            all_passed = False
            
    print("=" * 60, flush=True)
    if all_passed:
        print("RESULT: PASS - All services are operational!", flush=True)
        sys.exit(0)
    else:
        print("RESULT: FAIL - One or more services failed connection checks.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
