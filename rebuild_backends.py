"""
Medical Records RAG Demo — Rebuild GraphRAG & PageIndex Backends

Wipes existing Neo4j knowledge graph nodes and PageIndex JSON trees,
then rebuilds both from the existing OCR, reports, chunks, and normalized entities.
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from graph_backend.build import get_driver, build_all_patients
from pageindex_backend.build import build_all_pageindexes


def wipe_neo4j():
    """Wipes all nodes and relationships from the Neo4j database."""
    print("\n[1/4] 🧹 Wiping Neo4j Knowledge Graph...")
    driver = get_driver()
    try:
        with driver.session() as session:
            # Query count before wipe
            res = session.run("MATCH (n) RETURN count(n) AS node_count")
            record = res.single()
            count_before = record["node_count"] if record else 0
            print(f"      Found {count_before} nodes in Neo4j. Deleting...")

            session.run("MATCH (n) DETACH DELETE n")
            print("      ✓ Neo4j database successfully wiped clean.")
    finally:
        driver.close()


def wipe_pageindex():
    """Deletes existing PageIndex tree JSON files."""
    print("\n[2/4] 🧹 Wiping PageIndex local cache...")
    pageindex_dir = _ROOT / "data" / "pageindex"
    if pageindex_dir.exists():
        deleted = 0
        for f in pageindex_dir.glob("*.json"):
            f.unlink()
            deleted += 1
        print(f"      ✓ Deleted {deleted} PageIndex JSON file(s) from {pageindex_dir}.")
    else:
        pageindex_dir.mkdir(parents=True, exist_ok=True)
        print(f"      Created directory {pageindex_dir}.")


def rebuild_graph():
    """Rebuilds the Neo4j Knowledge Graph from existing chunks."""
    print("\n[3/4] 🏗️  Rebuilding GraphRAG (Neo4j)...")
    t0 = time.time()
    driver = get_driver()
    try:
        stats = build_all_patients(driver=driver)
    finally:
        driver.close()
    dt = round(time.time() - t0, 2)
    print(f"      ✓ GraphRAG build completed in {dt}s.")
    return stats


def rebuild_pageindex():
    """Rebuilds the PageIndex trees from existing reports/chunks."""
    print("\n[4/4] 🏗️  Rebuilding PageIndex Trees...")
    t0 = time.time()
    results = build_all_pageindexes()
    dt = round(time.time() - t0, 2)
    print(f"      ✓ PageIndex build completed in {dt}s.")
    return results


def main():
    print("=" * 70)
    print("      REBUILDING GRAPHRAG & PAGEINDEX BACKENDS (PRESERVING CHUNKS)     ")
    print("=" * 70)
    total_start = time.time()

    # Step 1: Wipe Neo4j
    wipe_neo4j()

    # Step 2: Wipe PageIndex
    wipe_pageindex()

    # Step 3: Rebuild Neo4j Graph
    graph_stats = rebuild_graph()

    # Step 4: Rebuild PageIndex Trees
    pageindex_results = rebuild_pageindex()

    total_elapsed = round(time.time() - total_start, 2)
    print("\n" + "=" * 70)
    print(f"🎉 ALL BACKENDS SUCCESSFULLY WIPED AND REBUILT IN {total_elapsed}s!")
    print("=" * 70)


if __name__ == "__main__":
    main()
