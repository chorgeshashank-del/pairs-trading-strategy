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
- and a separate final evaluation period not used for strategy development.

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

In the final evaluation, the Engle–Granger strategy performed substantially better than the Distance / SSD strategy.

| Method | Net Return | Convergence Rate |
|---|---:|---:|
| Engle–Granger | +4.34% | 70% |
| Distance / SSD | -10.38% | 40% |

The results suggest that the stricter relationship test used by Engle–Granger produced fewer but higher-quality trading opportunities in this sample.

## Repository Structure

```text
pairs-trading-strategy/
│
├── src/       Python code for data construction, pair selection and backtesting
├── results/   Final trade ledger and method-comparison results
├── report/    Full research report
└── README.md
