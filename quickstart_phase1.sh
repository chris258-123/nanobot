#!/bin/bash
# Quick start script for Phase 1 testing

set -e

echo "=========================================="
echo "Phase 1: Foundation - Quick Start"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Start Docker services
echo "Starting Docker services..."
cd /home/chris/Desktop/my_workspace/nanobot
docker-compose up -d

echo ""
echo "Waiting for services to start (30 seconds)..."
sleep 30

# Check Neo4j
echo ""
echo "Checking Neo4j..."
if curl -s http://localhost:7474 > /dev/null; then
    echo "✅ Neo4j is running (http://localhost:7474)"
else
    echo "⚠️  Neo4j may not be ready yet. Wait a bit longer."
fi

# Check Qdrant
echo ""
echo "Checking Qdrant..."
if curl -s http://localhost:6333/collections > /dev/null; then
    echo "✅ Qdrant is running (http://localhost:6333)"
else
    echo "⚠️  Qdrant may not be ready yet."
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -q -r requirements-novel.txt
echo "✅ Dependencies installed"

# Run test
echo ""
echo "=========================================="
echo "Running Phase 1 Test"
echo "=========================================="
echo ""

cd nanobot/skills/novel-workflow/scripts
python test_phase1.py

echo ""
echo "=========================================="
echo "Quick Start Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Open Neo4j browser: http://localhost:7474"
echo "   Login: neo4j / novel123"
echo ""
echo "2. Run Cypher queries:"
echo "   MATCH (e:Entity) RETURN e LIMIT 25"
echo "   MATCH (a:Character)-[r:RELATES]->(b:Character) RETURN a, r, b"
echo ""
echo "3. Check Canon DB:"
echo "   sqlite3 ~/.nanobot/workspace/canon_v2_test.db"
echo "   SELECT * FROM entity_registry;"
echo ""
echo "4. Read PHASE1_README.md for detailed documentation"
echo ""
