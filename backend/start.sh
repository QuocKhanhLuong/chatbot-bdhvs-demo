#!/bin/bash
# Start Python Backend Server

echo "🚀 Starting Python FastAPI Backend..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from example..."
    cp .env.example .env
    echo "❗ Please edit backend/.env and add your GOOGLE_API_KEY"
    exit 1
fi

# Start server
echo ""
echo "✅ Starting FastAPI server on http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""

cd api
python api.py
