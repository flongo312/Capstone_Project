# Frank Longo Optimal Investments for Home Savings Accumulation Capstone Project

## Full Project Compilation Instructions

To compile the full project, use the provided shell script DoWork.sh located at the base of the FLONGO folder. The script automates the following steps:

1. Determines and sets the script's directory as the base directory.
2. Lists the initial contents of the base directory.
3. Creates and activates a Python virtual environment in the base directory.
4. Navigates to the `Code` directory and sequentially runs:
     - `yfinance_data.py`: Fetches financial data using the yfinance library.
     - `init_filtering.py`: Processes and filters the initial dataset and calculates Capital Asset Pricing Model (CAPM), Sharpe 
     Ratio, and combined score method.
     - `mpt.py`: Implements the Modern Portfolio Theory (MPT) calculations.
     - `hindsight_data.py`: Gathers and processes data for hindsight analysis.
     - `hindsight.py`: Performs hindsight analysis on the processed data.
     - `mcs.py`: Executes Monte Carlo simulations for financial forecasting.
5. Lists the contents of the base directory post-script execution.
6. Compile LaTeX Documents:
   - Navigates to the `Paper` directory and compiles `Paper.tex` twice.
   - Checks for the successful creation of `Paper.pdf`.
   - Navigates to the `Slides` directory and compiles `slides.tex` twice.
   - Checks for the successful creation of `slides.pdf`.
   
   
### Running the Script

Ensure you have the necessary LaTeX distribution installed (e.g., TeX Live or MikTeX). Open a terminal, navigate to the directory containing the script, and run:

```sh
./DoWork.sh



###################################################################################################



## Paper Compilation Instructions

To compile the final paper document, use the provided shell script Compile.sh located at FLONGO/Paper. The script automates the following steps:

1. Determines and sets the script's directory as the working directory.
2. Defines the main LaTeX file to be compiled (named `Paper.tex`) and all other .tex files that feed into Paper.tex.
3. Runs `pdflatex` to generate initial auxiliary files.
4. Processes the bibliography using `bibtex`.
5. Runs `pdflatex` twice more to ensure all references and citations are updated.

### Running the Script

Ensure you have the necessary LaTeX distribution installed (e.g., TeX Live or MikTeX). Open a terminal, navigate to the directory containing the script, and run:

```sh
./Compile.sh
