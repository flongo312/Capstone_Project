#!/bin/bash

#########################################################
# Build the pdf Document with LaTeX
#########################################################

echo "#-------------------------------------------------"
echo ""
echo "Building the pdf document with LaTeX..."
echo ""

cd Paper

# Use pdflatex to compile the LaTeX file
pdflatex Outline
# Run LaTeX twice to fix references in the document
pdflatex Outline
# Use bibtex to compile the references
bibtex Outline
# Use pdflatex to compile the LaTeX file
pdflatex Outline
# Just to be sure
pdflatex Outline

cd ..

echo ""
echo "Finished building the pdf document with LaTeX."
echo ""
echo "#-------------------------------------------------"
echo ""

echo "#-------------------------------------------------"
echo "Moving Paper/Outline.pdf to FLongoOutline.pdf..."

mv Paper/Outline.pdf FLongoOutline.pdf

echo "Done."
echo "#-------------------------------------------------"

#########################################################
# End
#########################################################
