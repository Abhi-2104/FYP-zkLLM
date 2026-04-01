#!/bin/bash

# VerifAI Startup Script

echo "================================================================"
echo "  ____   ____           _  __    _    ___ "
echo " |  _ \\ / ___|   _ _ __(_)/ _|  / \\  |_ _|"
echo " | | | | |  _  | '__| | | |_  / _ \\  | | "
echo " | |_| | |_| | | |  | | |  _|/ ___ \\ | | "
echo " |____/ \\____| |_|  |_|_|_| /_/   \\_\\___|"
echo ""
echo " Zero-Knowledge LLM Verification System"
echo "================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found"
    echo "Please run this script from the user_interface directory"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "================================================================"
echo " VerifAI Server Starting..."
echo "================================================================"
echo ""
echo "🌐 Access the UI at:"
echo "   • Dashboard:  http://localhost:5000"
echo "   • Provider:   http://localhost:5000/provider"
echo "   • User:       http://localhost:5000/user"
echo "   • Verifier:   http://localhost:5000/verifier"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================================"
echo ""

# Start the Flask application
python3 app.py
