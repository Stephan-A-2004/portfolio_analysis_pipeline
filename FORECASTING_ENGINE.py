"""
Forecasting Engine

Implements expected-return estimators and asset return forecasting.
This module is called by the GUI when the user uploads a CSV file and writes
forecast results to a different CSV file.

Backtesting design:
- current forecast section from latest available training data
- one optional backtest window, controlled by a value calculated in 2B
- the backtest uses a fixed 1-year target horizon
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import yfinance as yf
from numpy.linalg import inv, pinv


def portfolio_analyser(
    lookback_period: str,
    tickers: List[str],
) -> Tuple[str, List[str]]:
    # ------------------------------------------------------------------
    # 0. Numerical helpers / estimators
    # ------------------------------------------------------------------

    def safe_scalar_div(num: float, denom: float) -> float:
        if not np.isfinite(num) or not np.isfinite(denom) or np.isclose(denom, 0.0):
            return np.nan
        return num / denom

    def annualise_from_daily(daily_return: float, trading_days: int = 252) -> float:
        if not np.isfinite(daily_return) or daily_return <= -1:
            return np.nan
        return (1 + daily_return) ** trading_days - 1

    def cumulative_from_daily(daily_return: float, horizon_days: int) -> float:
        if not np.isfinite(daily_return) or daily_return <= -1 or horizon_days < 1:
            return np.nan
        return (1 + daily_return) ** horizon_days - 1

    def sample_mean(x: np.ndarray) -> np.ndarray:
        return np.mean(x, axis=0)


    def St_MSh_estimator(x):
        Tn, N_T = x.shape
        x_bar = np.mean(x, axis=0)
        a1T_hat = np.sum(x_bar**2) / N_T
        trace_Sn = np.trace(np.cov(x, rowvar=False))
        a2T_hat = a1T_hat + trace_Sn / (Tn * N_T)
        c_star_hat = a1T_hat / a2T_hat
        theta_St_MSh = c_star_hat * x_bar
        return theta_St_MSh

    def D_MSh_estimator(x):
        Tn, N_T = x.shape
        x_bar = np.mean(x, axis=0)
        Sn = np.cov(x, rowvar=False)
        c_star_hat = (x_bar**2) / (x_bar**2 + (1 / Tn) * np.diag(Sn))
        theta_D_MSh = c_star_hat * x_bar
        return theta_D_MSh

    def O_LSh_estimator(x):
        Tn, N_T = x.shape
        x_bar = np.mean(x, axis=0)
        a1T_hat = np.sum(x_bar**2) / N_T
        trace_Sn = np.trace(np.cov(x, rowvar=False))
        a2T_hat = a1T_hat + trace_Sn / (Tn * N_T)

        s_i = np.sum(x, axis=1)
        term1 = np.sum(s_i**2)
        term2 = (np.sum(s_i)**2 - term1) / (Tn - 1)
        d2_hat = (term1 - term2) / (Tn**2 * N_T**2)
        grand_mean = np.mean(x_bar)
        d3_hat = np.sum((x_bar - grand_mean)**2) / N_T

        tilde_a1_hat = a2T_hat - a1T_hat - d2_hat
        tilde_a2_hat = tilde_a1_hat + d3_hat
        delta_star_hat = tilde_a1_hat / tilde_a2_hat

        theta_O_LSh = (1 - delta_star_hat) * x_bar + delta_star_hat * grand_mean
        return theta_O_LSh

    def T_LSh_estimator(x):
        Tn, N_T = x.shape
        x_bar = np.mean(x, axis=0)
        a1T_hat = np.sum(x_bar**2) / N_T
        trace_Sn = np.trace(np.cov(x, rowvar=False))
        a2T_hat = a1T_hat + trace_Sn / (Tn * N_T)

        s_i = np.sum(x, axis=1)
        term1 = np.sum(s_i**2)
        term2 = (np.sum(s_i)**2 - term1) / (Tn - 1)
        d2_hat = (term1 - term2) / (Tn**2 * N_T**2)
        grand_mean = np.mean(x_bar)
        d3_hat = np.sum((x_bar - grand_mean)**2) / N_T
        d4_hat = grand_mean**2

        tilde_a1_hat = a2T_hat - a1T_hat - d2_hat
        tilde_a2_hat = tilde_a1_hat + d3_hat
        delta_star_hat = tilde_a1_hat / tilde_a2_hat
        xi_star_hat = 1 - (d2_hat / (d2_hat + d4_hat)) / delta_star_hat

        theta_T_LSh = (1 - delta_star_hat) * x_bar + delta_star_hat * xi_star_hat * grand_mean
        return theta_T_LSh


    def Wang_estimator(x):
        n, p = x.shape
        x_bar = np.mean(x, axis=0)
        col_sum = np.sum(x, axis=0)
        total_sq_col = np.sum(col_sum**2)
        sum_sq_x = np.sum(x**2)
        Y1n = (total_sq_col - sum_sq_x) / (p * (n - 1))
        Y2n = (sum_sq_x - p * Y1n) / (n * p)
        row_sum = np.sum(x, axis=1)
        total_row = np.sum(row_sum)
        Y3n = (total_row**2 - np.sum(row_sum**2)) / (p**2 * (n - 1))
        Y4n = total_row / (n * p)

        denom = Y1n + Y2n - Y3n
        alpha_star = (Y1n - Y3n) / denom
        beta_star = (Y2n * Y4n) / denom

        theta_hat_Wang = alpha_star * x_bar + beta_star
        return theta_hat_Wang

    def BOP_estimator(x):
        n, p = x.shape
        y_bar = np.mean(x, axis=0)
        S_n = np.cov(x, rowvar=False)
        S_n_inv = pinv(S_n) if p > n else inv(S_n)
        mu_0 = np.ones(p)

        yS_inv_y = y_bar @ S_n_inv @ y_bar
        mu0_S_inv_mu0 = mu_0 @ S_n_inv @ mu_0
        yS_inv_mu0 = y_bar @ S_n_inv @ mu_0

        if p > n:
            term1 = yS_inv_y - n / (p - n)
        else:
            term1 = yS_inv_y - p / (n - p)
        term2 = yS_inv_y * mu0_S_inv_mu0

        alpha_mean = (term1 * mu0_S_inv_mu0 - yS_inv_mu0**2) / (term2 - yS_inv_mu0**2)
        beta_mean = (1 - alpha_mean) * yS_inv_mu0 / mu0_S_inv_mu0

        theta_hat_BOP = alpha_mean * y_bar + beta_mean * mu_0
        return theta_hat_BOP

    def CW_estimator(x):
        n, p = x.shape
        y_bar = np.mean(x, axis=0)
        S_n = np.cov(x, rowvar=False)
        S_n_plus = pinv(S_n)

        a = (n - 3) / (p - n + 4)
        scalar = y_bar @ S_n_plus @ y_bar
        term2 = max(1 - (a / scalar), 0)

        theta_hat_CW = (np.eye(p) - S_n @ S_n_plus) @ y_bar + term2 * (S_n @ S_n_plus @ y_bar)
        return theta_hat_CW

    def Jorion_estimator(x):
        Tn, N_T = x.shape
        mu_MLE = np.mean(x, axis=0)
        Sn = ((Tn - 1) / (Tn - N_T - 2)) * np.cov(x, rowvar=False)
        Sn_inv = inv(Sn)
        ones = np.ones(N_T)

        num = ones @ Sn_inv @ mu_MLE
        denom = ones @ Sn_inv @ ones
        mu0 = num / denom

        diff = mu_MLE - mu0 * ones
        d = Tn * (diff @ Sn_inv @ diff)
        w = (N_T + 2) / (N_T + 2 + d)

        theta_hat_Jorion = (1 - w) * mu_MLE + w * mu0 * np.ones(N_T)
        return theta_hat_Jorion

        # Python code for implementation of these estimators adopted from open-source repository.
        
    def run_model(model_name: str, data: np.ndarray) -> Optional[np.ndarray]:
        try:
            if model_name == "Sample_Mean":
                return sample_mean(data)
            if model_name == "St_MSh":
                return St_MSh_estimator(data)
            if model_name == "D_MSh":
                return D_MSh_estimator(data)
            if model_name == "O_LSh":
                return O_LSh_estimator(data)
            if model_name == "T_LSh":
                return T_LSh_estimator(data)
            if model_name == "Jorion":
                return Jorion_estimator(data)
            if model_name == "CW":
                return CW_estimator(data)
            if model_name == "BOP":
                return BOP_estimator(data)
            if model_name == "Wang":
                return Wang_estimator(data)
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------

    if not tickers:
        return "Error 2", []

    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        return "Error 2", []

    try:
        lookback_period_date = datetime.strptime(lookback_period, "%Y-%m-%d")
    except ValueError:
        return "Error 4", []

    requested_estimators = ["O_LSh", "Wang"] # Estimators used are decided based on this list. The list can be modifed to use other estimators in the future.

    # ------------------------------------------------------------------
    # 2. Download prices
    # ------------------------------------------------------------------

    real_today = datetime.fromtimestamp(time.time())
    download_end = (real_today + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        raw = yf.download(
            tickers,
            start=lookback_period_date,
            end=download_end,
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return "Error 1", []

    if raw.empty:
        return "Error 1", []

    try:
        prices = raw["Adj Close"]
    except Exception:
        return "Error 1", []

    if getattr(prices, "ndim", None) == 1:
        prices = prices.to_frame(name=tickers[0])

    if prices.empty:
        return "Error 1", []

    prices = prices.sort_index()
    prices = prices.dropna(axis=1, how="all")

    valid_downloaded_tickers = prices.columns.tolist()
    if not valid_downloaded_tickers:
        return "Error 2", []

    invalid_tickers = [t for t in tickers if t not in valid_downloaded_tickers]

    # ------------------------------------------------------------------
    # 2B. Determine backtest cutoff date
    # ------------------------------------------------------------------

    backtest_cutoff_date = None
    fixed_backtest_horizon_days = 252

    available_dates = prices.index.sort_values()

    # Need at least 1 training price row and 252 later trading dates
    cutoff_position = len(available_dates) - fixed_backtest_horizon_days - 1

    if cutoff_position >= 0:
        backtest_cutoff_date = available_dates[cutoff_position].to_pydatetime()

    if backtest_cutoff_date is not None and backtest_cutoff_date < lookback_period_date:
        backtest_cutoff_date = None

    # ------------------------------------------------------------------
    # 3. Clean universe for current forecast
    # ------------------------------------------------------------------

    full_training_returns = prices.pct_change(fill_method=None).dropna()
    full_training_returns = full_training_returns.dropna(axis=1)

    if full_training_returns.empty:
        return "Error 3", invalid_tickers

    tickers_used = list(full_training_returns.columns)
    dropped_after_cleaning = [t for t in valid_downloaded_tickers if t not in tickers_used]
    invalid_tickers_extended = list(dict.fromkeys(invalid_tickers + dropped_after_cleaning))

    x_full = full_training_returns.to_numpy()
    Tn_full, N_T_full = x_full.shape

    if Tn_full < 2 or N_T_full < 1:
        return "Error 3", invalid_tickers_extended

    # ------------------------------------------------------------------
    # 4. Current forecast from latest available data
    # ------------------------------------------------------------------

    model_names: List[str] = []
    daily_expected_returns: List[np.ndarray] = []

    for model_name in requested_estimators:
        theta_hat = run_model(model_name, x_full)

        if theta_hat is None:
            continue
        if len(theta_hat) != len(tickers_used):
            continue
        if np.any(~np.isfinite(theta_hat)):
            continue

        model_names.append(model_name)
        daily_expected_returns.append(theta_hat)

    if not model_names:
        return "Error 3", invalid_tickers_extended

    forecast_days = 252
    expected_returns_all_models: Dict[str, Dict[str, float]] = {}

    for model_name, mu_daily in zip(model_names, daily_expected_returns):
        expected_returns_all_models[model_name] = {
            ticker: annualise_from_daily(daily_return, forecast_days)
            for ticker, daily_return in zip(tickers_used, mu_daily)
        }

    # ------------------------------------------------------------------
    # 5. Optional single-window backtest (fixed 1-year horizon), requires lookback period of 2 years ago minimum)
    # ------------------------------------------------------------------

    min_training_returns_required = 504

    backtest_available = False 
    backtest_metadata: Dict[str, object] = {}
    backtest_detail_rows: List[Dict[str, object]] = []
    performance_summary: List[Dict[str, float]] = []

    if backtest_cutoff_date is not None:

        #print("\n--- BACKTEST ENTRY ---")
        #print("backtest_cutoff_date:", backtest_cutoff_date)

        train_prices_bt = prices.loc[prices.index <= backtest_cutoff_date, tickers_used]
        test_prices_bt = prices.loc[prices.index > backtest_cutoff_date, tickers_used]

        #print("\n--- DATA SPLIT ---")
        #print("train rows:", len(train_prices_bt))
        #print("test rows:", len(test_prices_bt))

        if not train_prices_bt.empty and not test_prices_bt.empty:
            train_returns_bt = train_prices_bt.pct_change(fill_method=None).dropna()
            train_returns_bt = train_returns_bt.dropna(axis=1)

            #print("\n--- TRAIN RETURNS ---")
            #print("Tn_bt (training observations):", len(train_returns_bt))
            #print("N_T_bt (assets):", train_returns_bt.shape[1] if not train_returns_bt.empty else 0)

            if not train_returns_bt.empty:
                tickers_bt = list(train_returns_bt.columns)

                if tickers_bt:
                    test_prices_bt = test_prices_bt[tickers_bt]

                    x_bt = train_returns_bt.to_numpy()
                    Tn_bt, N_T_bt = x_bt.shape
                    #print("\n--- PRE BACKTEST CHECK ---")
                    #print("Tn_bt:", Tn_bt)
                    #print("Required:", min_training_returns_required)
                    #print("Pass training condition:", Tn_bt >= min_training_returns_required)
                    if Tn_bt >= max(2, min_training_returns_required) and N_T_bt >= 1:
                        period_models: List[str] = []
                        period_daily_expected: List[np.ndarray] = []

                        for model_name in requested_estimators:
                            theta_hat = run_model(model_name, x_bt)

                            if theta_hat is None:
                                continue
                            if len(theta_hat) != len(tickers_bt):
                                continue
                            if np.any(~np.isfinite(theta_hat)):
                                continue

                            period_models.append(model_name)
                            period_daily_expected.append(theta_hat)

                        if period_models:
                            actual_vec_bt = []
                            usable_asset_mask = []

                            # Define the fixed test horizon using the common post-cutoff panel
                            common_test_index = test_prices_bt.index

                            #print("\n--- FORWARD WINDOW ---")
                            #print("forward_rows:", len(common_test_index))
                            #print("required:", fixed_backtest_horizon_days)
                            #print("Pass forward condition:", len(common_test_index) >= fixed_backtest_horizon_days)

                            if len(common_test_index) < fixed_backtest_horizon_days:
                                actual_vec_bt = np.array([], dtype=float)
                                usable_asset_mask = []
                            else:
                                fixed_test_end_date = common_test_index[fixed_backtest_horizon_days - 1]

                                for ticker in tickers_bt:
                                    train_series = train_prices_bt[ticker].dropna()

                                    # Need at least one known price on/before cutoff
                                    if len(train_series) < 1:
                                        actual_vec_bt.append(np.nan)
                                        usable_asset_mask.append(False)
                                        continue

                                    start_price = train_series.iloc[-1]

                                    # Use the common fixed-horizon end date, not per-ticker compressed indexing
                                    try:
                                        end_price = test_prices_bt.at[fixed_test_end_date, ticker]
                                    except KeyError:
                                        actual_vec_bt.append(np.nan)
                                        usable_asset_mask.append(False)
                                        continue

                                    if (
                                        not np.isfinite(start_price)
                                        or not np.isfinite(end_price)
                                        or start_price <= 0
                                        or end_price <= 0
                                    ):
                                        actual_vec_bt.append(np.nan)
                                        usable_asset_mask.append(False)
                                        continue

                                    realised_return = (end_price / start_price) - 1

                                    actual_vec_bt.append(realised_return)
                                    usable_asset_mask.append(True)

                                actual_vec_bt = np.array(actual_vec_bt, dtype=float)

                            if len(actual_vec_bt) > 0 and any(usable_asset_mask):
                                backtest_available = True
                                backtest_metadata = {
                                    "Cutoff Date": backtest_cutoff_date.strftime("%Y-%m-%d"),
                                    "Training Return Observations": Tn_bt,
                                    "Assets in Backtest": len(tickers_bt),
                                    "Fixed Test Horizon (Trading Days)": fixed_backtest_horizon_days,
                                    "Fixed Test End Date": fixed_test_end_date.strftime("%Y-%m-%d"),
                                    "Summary Includes Only Assets With Full Horizon": "Yes",
                                }

                                summary_errors_by_model: Dict[str, List[float]] = {}
                                summary_direction_by_model: Dict[str, List[bool]] = {}

                                for model_name, mu_daily in zip(period_models, period_daily_expected):
                                    predicted_vec_bt = np.array([
                                        cumulative_from_daily(mu_daily[i], fixed_backtest_horizon_days)
                                        if usable_asset_mask[i] else np.nan
                                        for i in range(len(tickers_bt))
                                    ], dtype=float)

                                    error_vec_bt = predicted_vec_bt - actual_vec_bt

                                    for i, ticker in enumerate(tickers_bt):
                                        row = {
                                            "Ticker": ticker,
                                            "Model": model_name,
                                            "Training Return Observations": Tn_bt,
                                            "Test Return Observations": fixed_backtest_horizon_days,
                                            "Included in Summary": (
                                                "Yes"
                                                if usable_asset_mask[i]
                                                and np.isfinite(predicted_vec_bt[i])
                                                and np.isfinite(actual_vec_bt[i])
                                                and np.isfinite(error_vec_bt[i])
                                                else "No"
                                            ),
                                            "Forecast Return": predicted_vec_bt[i],
                                            "Actual Return": actual_vec_bt[i],
                                            "Error": error_vec_bt[i],
                                        }
                                        backtest_detail_rows.append(row)

                                        if row["Included in Summary"] == "Yes":
                                            summary_errors_by_model.setdefault(model_name, []).append(float(error_vec_bt[i]))

                                            direction_correct = (
                                                (predicted_vec_bt[i] > 0 and actual_vec_bt[i] > 0)
                                                or
                                                (predicted_vec_bt[i] < 0 and actual_vec_bt[i] < 0)
                                            )
                                            summary_direction_by_model.setdefault(model_name, []).append(direction_correct)

                                for model_name, errs in summary_errors_by_model.items():
                                    if errs:
                                        err_arr = np.array(errs, dtype=float)
                                        direction_arr = np.array(summary_direction_by_model.get(model_name, []), dtype=bool)

                                        performance_summary.append({
                                            "Model": model_name,
                                            "MAE": float(np.mean(np.abs(err_arr))),
                                            "RMSE": float(np.sqrt(np.mean(err_arr ** 2))),
                                            "Bias": float(np.mean(err_arr)),
                                            "Directional Accuracy": float(np.mean(direction_arr)) if len(direction_arr) > 0 else np.nan,
                                            "Observations": int(len(err_arr)),
                                        })

                                performance_summary.sort(key=lambda row: row["RMSE"])

    # ------------------------------------------------------------------
    # 6. Write CSV
    # ------------------------------------------------------------------

    suggested_lookback_date = None

    if not backtest_available:
        # Suggest a date ~3 years back from today (757 trading days is approximately equal to 1097 calendar days, but to ensure the suggested date gurantees a backtest running, difference of 1105 days will be used as suggestion)
        suggested_lookback_date = real_today - timedelta(days=1105)

    output_file_name = f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_file_path = os.path.abspath(output_file_name)

    with open(output_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["FORECASTING ENGINE OUTPUT"])
        writer.writerow(["Generated at", datetime.now().isoformat()])
        writer.writerow(["Lookback period start", lookback_period])
        failed_estimators = [m for m in requested_estimators if m not in model_names]
        writer.writerow(["Estimators that failed or were skipped", ", ".join(failed_estimators) if failed_estimators else "None"])
        writer.writerow(["Requested tickers", ", ".join(tickers)])
        writer.writerow(["Analysed tickers", ", ".join(tickers_used)])
        writer.writerow(["Excluded / invalid tickers", ", ".join(invalid_tickers_extended) if invalid_tickers_extended else "None"])
        writer.writerow([])

        writer.writerow(["SECTION 1: CURRENT FORECAST OUTPUT (ANNUALISED TO 252 TRADING DAYS)"])
        forecast_header = ["Ticker"] + [f"Forecast {model_name} (%)" for model_name in model_names]
        writer.writerow(forecast_header)

        for ticker in tickers_used:
            row = [ticker]
            for model_name in model_names:
                val = expected_returns_all_models[model_name].get(ticker, np.nan)
                row.append(round(val * 100, 2) if np.isfinite(val) else "N/A")
            writer.writerow(row)

        writer.writerow([])

        if backtest_available:
            writer.writerow(["SECTION 2: BACKTEST METADATA"])
            writer.writerow(["Cutoff Date", backtest_metadata["Cutoff Date"]])
            writer.writerow(["Training Return Observations", backtest_metadata["Training Return Observations"]])
            writer.writerow(["Assets in Backtest", backtest_metadata["Assets in Backtest"]])
            writer.writerow(["Fixed Test Horizon (Trading Days)", backtest_metadata["Fixed Test Horizon (Trading Days)"]])
            writer.writerow(["Fixed Test End Date", backtest_metadata["Fixed Test End Date"]])
            writer.writerow(["Summary Includes Only Assets With Full Horizon", backtest_metadata["Summary Includes Only Assets With Full Horizon"]])
            writer.writerow([])

            writer.writerow(["SECTION 3: BACKTEST DETAIL (FORECAST VS REALISED RETURNS)"])
            writer.writerow([
                "Ticker",
                "Model",
                "Training Return Observations",
                "Test Return Observations",
                "Included in Summary",
                "Forecast Return (%)",
                "Actual Return (%)",
                "Error (%)",
            ])

            for row in backtest_detail_rows:
                writer.writerow([
                    row["Ticker"],
                    row["Model"],
                    row["Training Return Observations"],
                    row["Test Return Observations"],
                    row["Included in Summary"],
                    round(float(row["Forecast Return"]) * 100, 2) if np.isfinite(row["Forecast Return"]) else "N/A",
                    round(float(row["Actual Return"]) * 100, 2) if np.isfinite(row["Actual Return"]) else "N/A",
                    round(float(row["Error"]) * 100, 2) if np.isfinite(row["Error"]) else "N/A",
                ])

            writer.writerow([])

            writer.writerow(["SECTION 4: BACKTEST SUMMARY"])
            writer.writerow(["Model", "MAE (%)", "RMSE (%)", "Bias (%)", "Directional Accuracy (%)", "Observations"])

            if performance_summary:
                for row in performance_summary:
                    writer.writerow([
                        row["Model"],
                        round(row["MAE"] * 100, 2),
                        round(row["RMSE"] * 100, 2),
                        round(row["Bias"] * 100, 2),
                        round(row["Directional Accuracy"] * 100, 2) if np.isfinite(row["Directional Accuracy"]) else "N/A",
                        row["Observations"],
                    ])
            else:
                writer.writerow(["No summary generated", "N/A", "N/A", "N/A", "N/A", "N/A"])
        else:
            writer.writerow(["SECTION 2: NO BACKTEST GENERATED"])

            if suggested_lookback_date is not None:
                reason_text = (
                    "Insufficient clean training data before cutoff or insufficient "
                    "252-trading-day forward data after cutoff. "
                    f"To enable backtesting, select a lookback period start date on or before "
                    f"{suggested_lookback_date.strftime('%Y-%m-%d')}."
                )
            else:
                reason_text = (
                    "Insufficient clean training data before cutoff or insufficient "
                    "252-trading-day forward data after cutoff."
                )

            writer.writerow(["Reason", reason_text])

    return output_file_path, invalid_tickers_extended
