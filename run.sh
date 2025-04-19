#!/bin/bash

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Initialize the database
echo "Initializing the database..."
python3 init_db.py

# Run the app
echo "Starting FarmaBot..."
python3 app.py 