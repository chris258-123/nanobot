"""Parallel embedder with multi-threading and logging.

Embeds novel assets into Qdrant vector database using multiple workers.
"""

import os
# Disable SOCKS proxy at the very beginning
os.environ.pop('ALL_PROXY', None)
os.environ.pop('all_proxy', None)

import json
import httpx
from pathlib import Path
import argparse
from sentence_transformers import SentenceTransformer
try:
    from FlagEmbedding import FlagModel
    FLAG_MODEL_AVAILABLE = True
except ImportError:
    FLAG_MODEL_AVAILABLE = False
    print("Warning: FlagEmbedding not available, will use SentenceTransformer only")
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import logging
from datetime import datetime
import uuid


# Namespace for UUIDv5 generation (stable across runs)
ASSET_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "nanobot.novel.assets")


def stable_point_id(book_id: str, chapter: str, asset_type: str, asset_key: str) -> str:
    """Generate stable, deterministic point ID using UUIDv5.

    This ensures the same asset always gets the same ID across:
    - Different processes
    - Different runs
    - Different machines

    Args:
        book_id: Book identifier
        chapter: Chapter identifier
        asset_type: Type of asset (plot_beat, character_card, etc.)
        asset_key: Unique key for this asset (e.g., idx, entity_id, or content hash)

    Returns:
        Hex string UUID (32 characters)
    """
    composite_key = f"{book_id}|{chapter}|{asset_type}|{asset_key}"
    return uuid.uuid5(ASSET_NAMESPACE, composite_key).hex


# Thread-safe counters
stats_lock = Lock()
stats = {"processed": 0, "skipped": 0, "failed": 0, "total_points": 0}


def setup_logging(log_dir: str, book_id: str) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"embedder_{book_id}_{timestamp}.log"

    # Create logger
    logger = logging.getLogger("embedder")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to: {log_file}")
    return logger


def create_collection(qdrant_url: str, collection_name: str, vector_size: int, logger: logging.Logger):
    """Create Qdrant collection if it doesn't exist."""
    import os

    # Temporarily unset SOCKS proxy
    old_all_proxy = os.environ.pop('ALL_PROXY', None)
    old_all_proxy_lower = os.environ.pop('all_proxy', None)

    try:
        # Check if collection exists
        response = httpx.get(f"{qdrant_url}/collections/{collection_name}", timeout=10.0)
        if response.status_code == 200:
            logger.info(f"Collection '{collection_name}' already exists")
            return True
    except:
        pass

    # Create collection
    try:
        logger.info(f"Creating collection '{collection_name}' with vector size {vector_size}")
        response = httpx.put(
            f"{qdrant_url}/collections/{collection_name}",
            json={
                "vectors": {
                    "size": vector_size,
                    "distance": "Cosine"
                }
            },
            timeout=30.0
        )
        response.raise_for_status()
        logger.info(f"Collection '{collection_name}' created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return False
    finally:
        # Restore proxy settings
        if old_all_proxy:
            os.environ['ALL_PROXY'] = old_all_proxy
        if old_all_proxy_lower:
            os.environ['all_proxy'] = old_all_proxy_lower


def embed_asset_file(asset_file: Path, model, qdrant_url: str,
                     collection_name: str, logger: logging.Logger, worker_id: int,
                     use_flag_model: bool = False) -> tuple[bool, str, int]:
    """Embed a single asset file. Returns (success, message, points_count)."""
    import os

    # Temporarily unset SOCKS proxy
    old_all_proxy = os.environ.pop('ALL_PROXY', None)
    old_all_proxy_lower = os.environ.pop('all_proxy', None)

    try:
        # Load assets
        with open(asset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        book_id = data.get("book_id", "unknown")
        chapter = data.get("chapter", "unknown")

        points = []

        # Process each asset type
        asset_types = [
            ("plot_beats", "plot_beat"),
            ("character_cards", "character_card"),
            ("conflicts", "conflict"),
            ("settings", "setting"),
            ("themes", "theme"),
        ]

        for field_name, asset_type in asset_types:
            assets = data.get(field_name, [])
            if not isinstance(assets, list):
                continue

            for idx, asset in enumerate(assets):
                # Build text representation
                if asset_type == "plot_beat":
                    text = f"{asset.get('event', '')} {asset.get('impact', '')}"
                    characters = asset.get('characters', [])
                elif asset_type == "character_card":
                    text = f"{asset.get('name', '')}: {' '.join(asset.get('traits', []))} {asset.get('state', '')}"
                    characters = [asset.get('name', '')]
                elif asset_type == "conflict":
                    text = f"{asset.get('type', '')} {asset.get('description', '')}"
                    characters = asset.get('parties', [])
                elif asset_type == "setting":
                    text = f"{asset.get('location', '')} {asset.get('time', '')} {asset.get('atmosphere', '')}"
                    characters = []
                elif asset_type == "theme":
                    text = f"{asset.get('theme', '')} {asset.get('manifestation', '')}"
                    characters = []
                else:
                    continue

                if not text.strip():
                    continue

                # Generate embedding
                if use_flag_model:
                    # FlagModel returns numpy array, need to convert to list
                    embedding = model.encode([text])[0].tolist()
                else:
                    embedding = model.encode(text).tolist()

                # Create stable point ID using UUIDv5
                # Use idx as asset_key for list-based assets
                point_id = stable_point_id(book_id, chapter, asset_type, str(idx))

                points.append({
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "book_id": book_id,
                        "chapter": chapter,
                        "asset_type": asset_type,
                        "characters": characters if isinstance(characters, list) else [],
                        "text": text,
                        "metadata": asset
                    }
                })

        # Handle POV, tone, style (single objects)
        for field_name, asset_type in [("point_of_view", "point_of_view"), ("tone", "tone"), ("style", "style")]:
            asset = data.get(field_name, {})
            if not isinstance(asset, dict) or not asset:
                continue

            # Build text from dict values
            text = " ".join(str(v) for v in asset.values() if v)
            if not text.strip():
                continue

            if use_flag_model:
                embedding = model.encode([text])[0].tolist()
            else:
                embedding = model.encode(text).tolist()

            # Create stable point ID using UUIDv5
            # For single-object assets, use asset_type as key (no idx needed)
            point_id = stable_point_id(book_id, chapter, asset_type, "0")

            points.append({
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "book_id": book_id,
                    "chapter": chapter,
                    "asset_type": asset_type,
                    "characters": [],
                    "text": text,
                    "metadata": asset
                }
            })

        if not points:
            with stats_lock:
                stats["skipped"] += 1
            return True, f"[Worker {worker_id}] Skipped {asset_file.name} (no valid assets)", 0

        # Batch upsert to Qdrant
        response = httpx.put(
            f"{qdrant_url}/collections/{collection_name}/points",
            json={"points": points},
            timeout=60.0
        )
        response.raise_for_status()

        with stats_lock:
            stats["processed"] += 1
            stats["total_points"] += len(points)

        return True, f"[Worker {worker_id}] ✓ {asset_file.name} ({len(points)} points)", len(points)

    except Exception as e:
        with stats_lock:
            stats["failed"] += 1
        return False, f"[Worker {worker_id}] ✗ {asset_file.name}: {e}", 0
    finally:
        # Restore proxy settings
        if old_all_proxy:
            os.environ['ALL_PROXY'] = old_all_proxy
        if old_all_proxy_lower:
            os.environ['all_proxy'] = old_all_proxy_lower


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel asset embedding with logging")
    parser.add_argument("--assets-dir", required=True, help="Directory containing asset JSON files")
    parser.add_argument("--book-id", required=True, help="Book ID (for filtering files)")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--collection", default="novel_assets_v2", help="Collection name")
    parser.add_argument("--model", default="chinese",
                       choices=["chinese", "chinese-large", "multilingual", "multilingual-large"],
                       help="Embedding model")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers")
    parser.add_argument("--log-dir", default="/home/chris/Desktop/my_workspace/nanobot/logs",
                       help="Log directory")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_dir, args.book_id)

    # Model selection
    model_map = {
        "chinese": ("moka-ai/m3e-base", 768, False),
        "chinese-large": ("BAAI/bge-large-zh-v1.5", 1024, True),
        "multilingual": ("paraphrase-multilingual-MiniLM-L12-v2", 384, False),
        "multilingual-large": ("distiluse-base-multilingual-cased-v2", 512, False)
    }
    model_name, vector_size, use_flag_model = model_map[args.model]

    logger.info(f"Starting parallel embedding")
    logger.info(f"Assets directory: {args.assets_dir}")
    logger.info(f"Book ID: {args.book_id}")
    logger.info(f"Model: {model_name} ({vector_size}-dim)")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Collection: {args.collection}")

    # Create collection
    if not create_collection(args.qdrant_url, args.collection, vector_size, logger):
        logger.error("Failed to create collection, exiting")
        exit(1)

    # Load model
    logger.info(f"Loading model: {model_name}")
    if use_flag_model and FLAG_MODEL_AVAILABLE:
        logger.info("Using FlagModel (optimized for BGE models)")
        model = FlagModel(
            model_name,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=True
        )
    elif use_flag_model and not FLAG_MODEL_AVAILABLE:
        logger.warning("FlagModel requested but not available, falling back to SentenceTransformer")
        logger.warning("Note: Vector size may differ from expected")
        use_flag_model = False  # Disable flag for embedding
        model = SentenceTransformer(model_name)
    else:
        logger.info("Using SentenceTransformer")
        model = SentenceTransformer(model_name)
    logger.info("Model loaded successfully")

    # Get asset files
    assets_dir = Path(args.assets_dir)
    asset_files = sorted(assets_dir.glob(f"{args.book_id}_*_assets.json"))
    total = len(asset_files)

    if total == 0:
        logger.error(f"No asset files found matching pattern: {args.book_id}_*_assets.json")
        exit(1)

    logger.info(f"Found {total} asset files")
    logger.info("=" * 60)

    start_time = time.time()
    completed = 0

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                embed_asset_file,
                asset_file,
                model,
                args.qdrant_url,
                args.collection,
                logger,
                i % args.workers + 1,
                use_flag_model
            ): asset_file
            for i, asset_file in enumerate(asset_files)
        }

        # Process results as they complete
        for future in as_completed(future_to_file):
            completed += 1
            success, message, points_count = future.result()

            # Log every completion
            if not success or completed % 10 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0

                logger.info(f"[{completed}/{total}] {message}")
                logger.info(f"Progress: {stats['processed']} processed, {stats['skipped']} skipped, {stats['failed']} failed")
                logger.info(f"Total points: {stats['total_points']} | Speed: {rate:.2f} files/sec | ETA: {eta/60:.1f} min")
                logger.info("-" * 60)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Embedding complete!")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Total points embedded: {stats['total_points']}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Average speed: {total/elapsed:.2f} files/sec")
    logger.info("=" * 60)
