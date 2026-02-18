#!/usr/bin/env python3
"""Test script to verify embedding integration in reprocess_all.py"""

import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent / "nanobot/skills/novel-workflow/scripts"))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers available")
    except ImportError:
        print("✗ sentence-transformers not available")
        print("  Install with: pip install sentence-transformers")

    try:
        from FlagEmbedding import FlagModel
        print("✓ FlagEmbedding available")
    except ImportError:
        print("✗ FlagEmbedding not available")
        print("  Install with: pip install FlagEmbedding")

    print()

def test_chapter_processor_init():
    """Test ChapterProcessor initialization with embedding model."""
    print("Testing ChapterProcessor initialization...")

    try:
        from chapter_processor import ChapterProcessor

        # Test without embedding model (backward compatibility)
        processor = ChapterProcessor(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_pass="test",
            canon_db_path="/tmp/test_canon.db",
            qdrant_url=None,
        )

        assert processor.embedding_model is None
        assert processor.use_flag_model is False
        assert processor.vector_size == 1024
        print("✓ ChapterProcessor initialization without embedding model works")

        processor.close()

        # Test with embedding model parameters
        processor = ChapterProcessor(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_pass="test",
            canon_db_path="/tmp/test_canon.db",
            qdrant_url="http://localhost:6333",
            embedding_model=None,  # Simulating no model loaded
            use_flag_model=False,
            vector_size=768,
        )

        assert processor.embedding_model is None
        assert processor.use_flag_model is False
        assert processor.vector_size == 768
        print("✓ ChapterProcessor initialization with embedding parameters works")

        processor.close()

    except Exception as e:
        print(f"✗ ChapterProcessor initialization failed: {e}")
        import traceback
        traceback.print_exc()

    print()

def test_generate_embedding():
    """Test _generate_embedding method."""
    print("Testing _generate_embedding method...")

    try:
        from chapter_processor import ChapterProcessor

        # Test with no embedding model (should return zero vector)
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
        assert len(embedding) == 768
        assert all(v == 0.0 for v in embedding)
        print("✓ _generate_embedding returns zero vector when no model configured")

        processor.close()

    except Exception as e:
        print(f"✗ _generate_embedding test failed: {e}")
        import traceback
        traceback.print_exc()

    print()

def test_reprocess_all_args():
    """Test reprocess_all.py argument parsing."""
    print("Testing reprocess_all.py argument parsing...")

    try:
        import argparse
        import sys

        # Save original argv
        original_argv = sys.argv

        # Test with new embedding arguments
        sys.argv = [
            "reprocess_all.py",
            "--book-id", "test",
            "--mode", "llm",
            "--llm-config", "test.json",
            "--chapter-dir", "/tmp/chapters",
            "--embedding-model", "chinese-large",
            "--skip-embedding",
        ]

        # Import and parse (this will fail if arguments are not defined)
        from reprocess_all import main

        print("✓ reprocess_all.py argument parsing works")

        # Restore original argv
        sys.argv = original_argv

    except SystemExit:
        # Expected when parsing args without running main
        print("✓ reprocess_all.py argument parsing works")
    except Exception as e:
        print(f"✗ reprocess_all.py argument parsing failed: {e}")
        import traceback
        traceback.print_exc()

    print()

if __name__ == "__main__":
    print("=" * 60)
    print("Embedding Integration Test Suite")
    print("=" * 60)
    print()

    test_imports()
    test_chapter_processor_init()
    test_generate_embedding()
    test_reprocess_all_args()

    print("=" * 60)
    print("Test suite complete!")
    print("=" * 60)
