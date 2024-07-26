#!/bin/bash

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the directory of the script
cd "$DIR"

# Name of your main .tex file (without the extension)
MAIN_FILE="Paper"

# Run pdflatex to generate the initial .aux file
pdflatex $MAIN_FILE.tex

# Run bibtex to process the bibliography
bibtex $MAIN_FILE

# Run pdflatex twice to ensure references are updated
pdflatex $MAIN_FILE.tex
pdflatex $MAIN_FILE.tex

echo "Compilation process finished."