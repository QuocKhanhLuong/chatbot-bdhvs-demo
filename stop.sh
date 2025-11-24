#!/bin/bash

# Stop all processes
echo "🛑 Stopping all services..."

# Kill backend
pkill -f "python api.py" 2>/dev/null && echo "✅ Backend stopped"

# Kill frontend
lsof -ti:3000 | xargs kill -9 2>/dev/null && echo "✅ Frontend stopped"

echo "✅ All services stopped"
