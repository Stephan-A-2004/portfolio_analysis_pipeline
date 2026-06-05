# Portfolio Analysis Pipeline

A desktop application that forecasts expected returns for a portfolio of stocks using statistical shrinkage estimators. Upload a CSV portfolio file, set a historical lookback window, and receive a detailed forecast (with an optional backtest) saved as a CSV.

---

## Requirements

- Python 3.11.0 (tested version)
- Dependencies: `numpy`, `yfinance`, `pandas`, `PySide6`

---

## How to Run

1. Run the application by executing `GUI.py`:

   ```bash
   python GUI.py
   ```

2. Set a **lookback period start date** using the date picker under *"Lookback period start date (Historical data window start date)"*. This is the earliest date from which historical price data will be downloaded.

3. Click **"Apply Time Period"** to confirm the date. This step is required before uploading a file.

4. Click **"Upload Portfolio File (must be CSV)"** and select your portfolio CSV file.

5. Wait for processing. A popup will appear confirming success and showing the output file path.

---

## Portfolio CSV Format

Your input CSV must contain a column of stock ticker symbols. The application auto-detects the ticker column by looking for a column where values are short, uppercase strings (up to 7 characters).

A sample portfolio file is provided in the `sample_portfolio_input/` folder for reference.

**Requirements:**
- Tickers must be written in **UPPERCASE** (e.g. `AAPL`, not `aapl`)
- Each ticker should appear **only once** in the file
- The file must be in **CSV format**

**Ticker format by region:**

| Region    | Format    | Example  |
|-----------|-----------|----------|
| USA       | No suffix | `AAPL`   |
| Japan     | `.T`      | `7203.T` |
| UK        | `.L`      | `VOD.L`  |
| Canada    | `.TO`     | `RY.TO`  |
| Germany   | `.DE`     | `BMW.DE` |
| France    | `.PA`     | `AIR.PA` |
| Hong Kong | `.HK`     | `0700.HK`|

> **Note:** Tickers with exchange suffixes (e.g. `.L`, `.T`) may work but are not guaranteed to be supported.

---

## Lookback Period Guidance

- The start date must be **between 1 January 2000 and today**
- A lookback period of **at least 2 years ago** is recommended for forecasting
- To enable backtesting, the application requires at least 504 trading days of training data before a cutoff, plus 252 trading days of data after it. If the lookback period is too short, the output file will suggest a suitable start date

---

## Output

The output is a timestamped CSV file (e.g. `portfolio_analysis_20260601_143022.csv`) saved in the **current working directory**, which is the folder from which you ran the script. The full path is printed to the console and shown in the success popup.

### Output File Structure

The CSV contains up to four sections:

**Section 1: Current Forecast**
Annualised expected returns (%) for each ticker, produced by two shrinkage estimators: `O_LSh` and `Wang`. Returns are annualised to 252 trading days.

**Section 2: Backtest Metadata** *(if backtest ran)*
Details of the backtest window: cutoff date, number of training return observations, test horizon, and end date.

**Section 3: Backtest Detail** *(if backtest ran)*
Per-ticker comparison of forecast vs. actual returns over the 252-trading-day test horizon, including error values.

**Section 4: Backtest Summary** *(if backtest ran)*
Aggregated model performance metrics: MAE, RMSE, Bias, and Directional Accuracy.

If the lookback period is too short for a backtest, Section 2 will explain why and suggest an appropriate start date.

---

## Forecasting Models

The application uses two **shrinkage mean estimators**, `O_LSh` and `Wang`, to produce expected daily returns, which are then annualised to 252 trading days.

---

## Settings

The application includes a Settings page accessible from the main menu, with options for:
- Light / Dark mode
- Button colour (blue, dark-blue, green)
- Text size (Small, Medium, Large, Extra Large)

Settings are saved to a `settings/config.json` file in the script directory.

---

## Error Messages

| Message | Cause |
|---|---|
| *"Please click the 'Apply Time Period' button before loading a portfolio."* | Date was not confirmed before uploading a file |
| *"No valid ticker column found..."* | CSV does not contain a recognisable uppercase ticker column |
| *"Could not fetch historical adjusted close price data..."* | Network error or failure to download price data |
| *"No valid tickers detected..."* | All tickers were invalid or returned no data |
| *"Please load the file again."* | An internal error occurred during forecasting |

---

## Notes

- Internet access is required to download historical price data via `yfinance`
- If some tickers are invalid or cannot be fetched, they will be listed in the success popup and in the output CSV header
- The application uses **adjusted close prices** for all calculations
