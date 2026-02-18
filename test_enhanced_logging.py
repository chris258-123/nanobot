#!/usr/bin/env python3
"""Test enhanced logging for embedding integration."""

import sys
import tempfile
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent / "nanobot/skills/novel-workflow/scripts"))

def test_logging():
    """Test that logging works correctly."""
    print("Testing enhanced logging...")
    print()

    try:
        from chapter_processor import ChapterProcessor

        # Create temporary log file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name

        print(f"Log file: {log_file}")
        print()

        # Test 1: ChapterProcessor with no embedding model
        print("Test 1: ChapterProcessor without embedding model")
        processor = ChapterProcessor(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_pass="test",
            canon_db_path="/tmp/test_canon.db",
            qdrant_url="http://localhost:6333",
            embedding_model=None,
            use_flag_model=False,
            vector_size=768,
        )
        print("✓ Initialization logged")
        processor.close()
        print()

        # Test 2: Generate embedding with no model
        print("Test 2: Generate embedding without model (should return zero vector)")
        processor = ChapterProcessor(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_pass="test",
            canon_db_path="/tmp/test_canon.db",
            qdrant_url="http://localhost:6333",
            embedding_model=None,
            use_flag_model=False,
            vector_size=768,
        )
        embedding = processor._generate_embedding("测试文本")
        print(f"✓ Generated embedding: {len(embedding)} dimensions")
        print(f"✓ All zeros: {all(v == 0.0 for v in embedding)}")
        processor.close()
        print()

        # Read log file
        print("Log file contents:")
        print("-" * 60)
        with open(log_file, 'r') as f:
            print(f.read())
        print("-" * 60)

        # Cleanup
        Path(log_file).unlink()

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Logging Test")
    print("=" * 60)
    print()

    test_logging()

    print()
    print("=" * 60)
    print("Test complete!")
    print("=" * 60)
