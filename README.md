# Statistical Pairs Trading on Indian Equities 

A research project comparing two statistical pairs-trading methods on Indian listed equities:

- **Distance / Minimum SSD** — selects pairs whose normalized price paths stay close together. SSD means *sum of squared differences* between the two normalized price series.
- **Engle–Granger (EG)** — selects pairs whose prices have a statistically stable long-run relationship, tested using cointegration.

The project covers the complete process from historical-universe construction and price adjustment to pair selection, trading rules, backtesting, transaction costs, and final method comparison.

## Objective

Test whether a simple relative-value pairs-trading strategy can produce meaningful results after accounting for:

- changes in the investable stock universe over time,
- dividends and other corporate actions,
- trading-calendar alignment,
- realistic short-selling eligibility,
- transaction costs,
- and a reserved final evaluation period, with its information-exposure limitations disclosed below.

The analysis uses Indian pharmaceutical equities and maintains a point-in-time universe so that stocks are included only when they were actually eligible at that date.

## Methodology

### 1. Data preparation

Historical equity data are cleaned and adjusted for corporate actions.

The pipeline also reconstructs:

- point-in-time index membership,
- historical F&O eligibility,
- the final investable universe,
- and data-quality checks.

### 2. Distance / Minimum SSD

Each stock-price series is normalized over the formation period.

For every candidate pair, the distance measure is

\[
SSD = \sum_t (A_t-B_t)^2
\]

where \(A_t\) and \(B_t\) are the normalized prices.

Pairs with the smallest SSD are considered the closest historical matches.

### 3. Engle–Granger

For each candidate pair, one stock is regressed on the other and the resulting spread is tested for stationarity.

A stationary spread is one that tends to return toward a stable level rather than drifting indefinitely.

Both regression directions are evaluated before selecting the final relationship.

### 4. Trading and backtesting

The selected pairs are tested using:

- a 12-month formation period,
- a 6-month trading period,
- spread-based entry and exit rules,
- realistic trade eligibility,
- and multiple transaction-cost assumptions.

The final comparison covers the period from February 2017 through July 2026.

## Main Result

Over the full comparison period, the Engle–Granger strategy outperformed the Distance / SSD strategy under the primary specification.

| Method | Net Return | Convergence Rate |
|---|---:|---:|
| Engle–Granger | +4.34% | 70% |
| Distance / SSD | -10.38% | 40% |

The results suggest that the stricter relationship test used by Engle–Granger produced fewer but higher-quality trading opportunities in this sample.


## Repository Structure

- `src/`: Python scripts for data preparation, pair selection, backtesting and figure generation.
- `results/`: Trade ledger, performance comparison, monthly returns, daily portfolio values, trade statistics, attribution and sensitivity results.
- `report/`: Research report.
- `exhibits/`: Cost-sensitivity and formation-window figures.
- `requirements.txt`: Specified Python package versions.

## Setup and Figure Reproduction

Download and extract the repository, then open a terminal in the folder containing `requirements.txt`.

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Recreate both figures:

```bash
python src/make_submission_exhibits.py
```

The command reads the saved result tables and writes:

- `exhibits/figure_1_cost_sensitivity.png`
- `exhibits/figure_2_formation_window_sensitivity.png`

Its input files are:

- `results/FINAL_COST_SENSITIVITY.csv`
- `results/ROBUSTNESS_COMPARISON.csv`
- `results/FULL_DEV_OOS_COMPARISON.csv`

The figure script sets the NumPy random seed to `20260830`. The plotted figures use saved results and do not require random sampling.

## Backtest Reproduction

Both backtests and the method-comparison script were successfully rerun from the prepared datasets and cached exchange reports included in this repository. Daily portfolio results matched the published outputs within numerical tolerance.

The reproduced cumulative net returns were:

- SSD: -10.38%
- Engle–Granger: +4.34%

### Run instructions

Install the packages listed in `requirements.txt` first.

Open Windows PowerShell in the repository's main folder—the folder containing `README.md` and `requirements.txt`.

Set the project directory:

```powershell
$env:PAIR_TRADING_PROJECT_ROOT = (Get-Location).Path
```

Run these commands in order. Wait for each command to finish successfully before running the next:

```powershell
python src/02_nifty_pharma_total_return_construction.py
python src/03_nifty_pharma_point_in_time_membership_v2.py
python src/engle_granger_pair_selection_BIDIRECTIONAL_FINAL.py
python src/ssd_COMPLETE_FASTTRACK_FINAL_WINDOW_FIXED.py
python src/engle_granger_COMPLETE_REQUIRED_BACKTEST_FIXED.py
python src/final_ssd_vs_eg_comparison.py
```

### Generated outputs

The scripts write regenerated outputs to:

- `pair_trading_methods/SSD/05_FINAL_FASTTRACK_REQUIRED/`
- `pair_trading_methods/ENGLE_GRANGER/01_pair_selection/`
- `pair_trading_methods/ENGLE_GRANGER/02_FINAL_BACKTEST/`
- `pair_trading_methods/FINAL_SSD_VS_EG_COMPARISON/`

The published tables in the top-level `results/` folder remain saved reference outputs; the backtest commands do not automatically replace them.

To regenerate the figures from those published reference tables:

```powershell
python src/make_submission_exhibits.py
```

### Scope of verification

Verification covered pair selection, both backtests, method comparison and figure generation using the packaged prepared data.

Rebuilding the prepared datasets from raw exchange downloads has not yet been verified from this repository alone. Benchmark returns and strategy returns are calculated between the same shared observation dates for both methods.

Total-return construction was also verified using the supplied cleaned equity data and corporate-action ledger. The rebuilt dataset matched the existing 68,309-row, 41-column dataset within numerical tolerance. This does not verify the earlier steps that produce the cleaned equity data, corporate-action ledger or investable universe from original sources.

Historical index membership reconstruction was also tested. The rebuilt members-only dataset matched the saved dataset: 36,646 rows and 50 columns. This step reconstructs index membership; the later liquidity and futures-eligibility filters remain separate.

## Results and Interpretation

The reported +4.34% Engle–Granger return and -10.38% SSD return are cumulative net returns over the full comparison period, February 2017 through July 2026. They are not annual returns or returns from the reserved final period alone.

Engle–Granger generated 10 trades and SSD generated 20 trades. Changing the formation-window length reversed the ranking between methods, so the primary comparison does not establish that Engle–Granger is universally superior.

## Evaluation Period and Limitations

The reserved final period begins on 1 August 2024. Earlier implementation diagnostics exposed some information from this period before the final specification was frozen. It is therefore a partially exposed evaluation period, not a completely untouched test.

Engle–Granger generated no trades in that period. Its 0% return represents remaining in cash and provides no evidence of profitable out-of-sample trading.

Other limitations include:

- Futures profit and loss uses an adjusted spot-price approximation rather than reconstructed historical futures contracts.
- Exact monthly futures rolls and whole-number contract sizing are not modelled.
- Historical surveillance restrictions and locked-circuit execution are not fully reconstructed.
- The benchmark lacks 11 strategy-calendar dates. Both methods compare returns between shared observation dates, so intervals spanning missing observations cover multiple sessions. Missing benchmark prices are not filled.
- The small trade sample limits confidence in the results.
