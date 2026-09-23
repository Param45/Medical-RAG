"""
Neo4j Connectivity Smoke Test for Medical RAG (Task 0.6)
Validates that the Neo4j instance is reachable with credentials from .env.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
dotenv_path = _ROOT_DIR / ".env"
if not dotenv_path.exists():
    print(f"[ERROR] .env file not found at: {dotenv_path}", file=sys.stderr)
    print("Please ensure .env is created and filled with credentials.", file=sys.stderr)
    sys.exit(1)

load_dotenv(dotenv_path)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_URI or not NEO4J_PASSWORD:
    print("[ERROR] NEO4J_URI or NEO4J_PASSWORD is not set in .env", file=sys.stderr)
    sys.exit(1)

try:
    from neo4j import GraphDatabase
except ImportError:
    print("[ERROR] 'neo4j' package is not installed in the current environment.", file=sys.stderr)
    sys.exit(1)


def test_neo4j_connection():
    print("=== Neo4j Connectivity Smoke Test (Task 0.6) ===")
    print(f"[*] Target URI:      {NEO4J_URI}")
    print(f"[*] Username:        {NEO4J_USERNAME}")
    print("[*] Connecting to Neo4j...")

    driver = None
    uris_to_try = [NEO4J_URI]
    if NEO4J_URI.startswith("neo4j+s://"):
        uris_to_try.append(NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://"))

    connected = False
    last_error = None

    for uri in uris_to_try:
        try:
            print(f"[*] Attempting connection with URI: {uri} ...")
            driver = GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            driver.verify_connectivity()
            
            with driver.session() as session:
                result = session.run("RETURN 1 AS ok")
                record = result.single()
                if record and record["ok"] == 1:
                    print(f"[*] Query executed successfully: RETURN 1 AS ok -> result: {record['ok']}")
                else:
                    print(f"[WARNING] Unexpected query result: {record}")

            print("\n" + "=" * 60)
            print("[SUCCESS] Neo4j instance is reachable and authenticated successfully!")
            print("=" * 60)
            connected = True
            break
        except Exception as e:
            last_error = e
            if driver:
                driver.close()
                driver = None

    if not connected:
        print("\n" + "=" * 60, file=sys.stderr)
        print(f"[FAILURE] Neo4j connection failed: {last_error}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)
    else:
        if driver:
            driver.close()
            print("[*] Driver connection closed cleanly.")



if __name__ == "__main__":
    test_neo4j_connection()
