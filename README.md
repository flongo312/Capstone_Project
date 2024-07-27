# Frank Longo Optimal Investments for Home Savings Accumulation Capstone Project

## Full Project Compilation Instructions

To compile the full project, use the provided shell script `DoWork.sh` located at the base of the FLONGO folder. The script automates the following steps:


1. Navigates to the `Code` directory and sequentially runs:
   - `yfinance_data.py`: Fetches financial data using the yfinance library.
   - `init_filtering.py`: Processes and filters the initial dataset and calculates Capital Asset Pricing Model (CAPM), Sharpe Ratio, and combined score method.
   - `mpt.py`: Implements the Modern Portfolio Theory (MPT) calculations.
   - `hindsight_data.py`: Gathers and processes data for hindsight analysis.
   - `hindsight.py`: Performs hindsight analysis on the processed data.
   - `mcs.py`: Executes Monte Carlo simulations for financial forecasting.
2. Lists the contents of the base directory post-script execution.
3. Compile LaTeX Documents:
   - Navigates to the `Paper` directory and compiles `Paper.tex` twice.
   - Checks for the successful creation of `Paper.pdf`.

### Running the Script

Ensure you have the necessary LaTeX distribution installed (e.g., TeX Live or MikTeX). Open a terminal, navigate to the directory containing the script, and run:

```sh
./DoWork.sh
