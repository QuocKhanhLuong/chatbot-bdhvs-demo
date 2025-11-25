#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${RED}🛑 Stopping Chatbot BDHVS services...${NC}"

# Kill processes on ports
lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null

# Kill by name
pkill -f "python -m app.main" 2>/dev/null
pkill -f "next dev" 2>/dev/null
pkill -f "uvicorn" 2>/dev/null

echo -e "${GREEN}✅ All services stopped${NC}"
