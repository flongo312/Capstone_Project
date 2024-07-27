#!/bin/bash

# Set the base directory to the absolute path of the script's directory
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "BASE_DIR is $BASE_DIR"

# List the contents of the base directory
echo "Listing contents of BASE_DIR:"
ls "$BASE_DIR"

# Navigate to the Code directory and run Python scripts
cd "$BASE_DIR/Code"

echo "Running yfinance_data.py..."
python3 yfinance_data.py || { echo "yfinance_data.py failed"; exit 1; }

echo "Running init_filtering.py..."
python3 init_filtering.py || { echo "init_filtering.py failed"; exit 1; }

echo "Running mpt.py..."
python3 mpt.py || { echo "mpt.py failed"; exit 1; }

echo "Running hindsight_data.py..."
python3 hindsight_data.py || { echo "hindsight_data.py failed"; exit 1; }

echo "Running hindsight.py..."
python3 hindsight.py || { echo "hindsight.py failed"; exit 1; }

echo "Running mcs.py..."
python3 mcs.py || { echo "mcs.py failed"; exit 1; }

# List the contents of the base directory again to verify
echo "Listing contents of BASE_DIR after running scripts:"
ls "$BASE_DIR"

# Navigate to the Paper directory and compile the LaTeX document
cd "$BASE_DIR/Paper" || { echo "Failed to navigate to $BASE_DIR/Paper"; exit 1; }
echo "Current directory: $(pwd)"

echo "Compiling Paper.tex..."
pdflatex Paper.tex || { echo "First compilation of Paper.tex failed"; exit 1; }
pdflatex Paper.tex || { echo "Second compilation of Paper.tex failed"; exit 1; }

# Check if the Paper.pdf was successfully created
if [ -f "Paper.pdf" ]; then
    echo "Paper compiled successfully."
else
    echo "Error compiling Paper.tex"
    exit 1
fi

echo "All scripts and LaTeX compilation executed successfully."

echo "DoWork.sh completed successfully."
