import os
from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================
# FINAL SSD vs ENGLE-GRANGER COMPARISON
# =============================================================================
#
# IMPORTANT:
# This script DOES NOT rerun either strategy.
# It only reads the frozen final SSD and EG outputs and places them side by side.
# =============================================================================

PROJECT_ROOT = Path(
    os.environ.get(
        "PAIR_TRADING_PROJECT_ROOT",
        r"C:\fin proj"
    )
)

SSD_ROOT = (
    PROJECT_ROOT
    / "pair_trading_methods"
    / "SSD"
    / "05_FINAL_FASTTRACK_REQUIRED"
)

SSD_RESULT_DIR = SSD_ROOT / "04_results"
SSD_AUDIT_DIR = SSD_ROOT / "05_audit"
SSD_SELECTION_DIR = SSD_ROOT / "01_pair_selection"

EG_ROOT = (
    PROJECT_ROOT
    / "pair_trading_methods"
    / "ENGLE_GRANGER"
    / "02_FINAL_BACKTEST"
)

EG_SELECTION_DIR = (
    PROJECT_ROOT
    / "pair_trading_methods"
    / "ENGLE_GRANGER"
    / "01_pair_selection"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "pair_trading_methods"
    / "FINAL_SSD_VS_EG_COMPARISON"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# -----------------------------------------------------------------------------
# Required frozen inputs
# -----------------------------------------------------------------------------

SSD_PERF = SSD_RESULT_DIR / "03_REQUIRED_PERFORMANCE_METRICS.csv"
SSD_TRADE_STATS = SSD_RESULT_DIR / "05_REQUIRED_TRADE_STATISTICS.csv"
SSD_CAPACITY = SSD_RESULT_DIR / "12_STRATEGY_CAPACITY_RUPEES.csv"
SSD_MARKET = SSD_RESULT_DIR / "13_NIFTY500_BETA_CORRELATION.csv"
SSD_ROBUST = SSD_RESULT_DIR / "14_REQUIRED_ROBUSTNESS_PERFORMANCE.csv"
SSD_SELECTION = SSD_SELECTION_DIR / "02_SELECTED_VALIDATED_SSD_PAIRS.csv"
SSD_AUDIT = SSD_AUDIT_DIR / "02_SSD_FINAL_FASTTRACK_AUDIT.csv"

EG_PERF = EG_ROOT / "05A_REQUIRED_PERFORMANCE_METRICS.csv"
EG_TRADE_STATS = EG_ROOT / "06A_REQUIRED_TRADE_STATISTICS.csv"
EG_CAPACITY = EG_ROOT / "09B_STRATEGY_CAPACITY_RUPEES.csv"
EG_MARKET = EG_ROOT / "10_NIFTY500_BETA_CORRELATION.csv"
EG_ROBUST = EG_ROOT / "13C_REQUIRED_ROBUSTNESS_PERFORMANCE.csv"
EG_SELECTION = EG_SELECTION_DIR / "03_SELECTED_ENGLE_GRANGER_PAIRS.csv"
EG_AUDIT = EG_ROOT / "11_EG_FINAL_BACKTEST_AUDIT.csv"

REQUIRED_FILES = [
    SSD_PERF,
    SSD_TRADE_STATS,
    SSD_CAPACITY,
    SSD_MARKET,
    SSD_ROBUST,
    SSD_SELECTION,
    SSD_AUDIT,
    EG_PERF,
    EG_TRADE_STATS,
    EG_CAPACITY,
    EG_MARKET,
    EG_ROBUST,
    EG_SELECTION,
    EG_AUDIT,
]

missing = [
    str(p)
    for p in REQUIRED_FILES
    if not p.exists()
]

if missing:
    raise FileNotFoundError(
        "Missing frozen comparison inputs:\n\n"
        + "\n".join(missing)
    )


def clean(df):
    out = df.copy()
    out.columns = (
        out.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.upper()
    )
    return out


def require_columns(df, cols, name):
    missing_cols = set(cols) - set(df.columns)
    if missing_cols:
        raise RuntimeError(
            f"{name} missing columns: {sorted(missing_cols)}"
        )


def one_row(df, **filters):
    x = df.copy()

    for col, value in filters.items():
        x = x[
            x[col].eq(value)
        ]

    if len(x) != 1:
        raise RuntimeError(
            f"Expected exactly one row for filters {filters}; found {len(x)}."
        )

    return x.iloc[0]


def pct(x):
    return (
        np.nan
        if pd.isna(x)
        else
        float(x)
    )


# =============================================================================
# 1. LOAD FROZEN OUTPUTS
# =============================================================================

ssd_perf = clean(pd.read_csv(SSD_PERF))
ssd_stats = clean(pd.read_csv(SSD_TRADE_STATS))
ssd_capacity = clean(pd.read_csv(SSD_CAPACITY))
ssd_market = clean(pd.read_csv(SSD_MARKET))
ssd_robust = clean(pd.read_csv(SSD_ROBUST))
ssd_selection = clean(pd.read_csv(SSD_SELECTION))
ssd_audit = clean(pd.read_csv(SSD_AUDIT))

eg_perf = clean(pd.read_csv(EG_PERF))
eg_stats = clean(pd.read_csv(EG_TRADE_STATS))
eg_capacity = clean(pd.read_csv(EG_CAPACITY))
eg_market = clean(pd.read_csv(EG_MARKET))
eg_robust = clean(pd.read_csv(EG_ROBUST))
eg_selection = clean(pd.read_csv(EG_SELECTION))
eg_audit = clean(pd.read_csv(EG_AUDIT))


require_columns(
    ssd_perf,
    [
        "COST_MULTIPLIER",
        "PERIOD",
        "START_DATE",
        "END_DATE",
        "TOTAL_RETURN",
        "CAGR",
        "ANNUALIZED_VOLATILITY",
        "SHARPE_RF0",
        "SORTINO_RF0",
        "MAX_DRAWDOWN",
        "CALMAR",
        "AVG_GROSS_EXPOSURE_TO_NAV",
        "AVG_ABS_NET_EXPOSURE_TO_NAV",
        "TOTAL_TURNOVER_TO_NAV",
    ],
    "SSD performance"
)

require_columns(
    eg_perf,
    [
        "COST_MULTIPLIER",
        "PERIOD",
        "START_DATE",
        "END_DATE",
        "TOTAL_RETURN",
        "CAGR",
        "ANNUALIZED_VOLATILITY",
        "SHARPE_RF0",
        "SORTINO_RF0",
        "MAX_DRAWDOWN",
        "CALMAR",
        "AVG_GROSS_EXPOSURE_TO_NAV",
        "AVG_ABS_NET_EXPOSURE_TO_NAV",
        "TOTAL_TURNOVER_TO_NAV",
    ],
    "EG performance"
)

require_columns(
    ssd_stats,
    [
        "PERIOD",
        "N_TRADES",
        "HIT_RATE",
        "HOLDING_DAYS_MEAN",
        "HOLDING_DAYS_MEDIAN",
        "CONVERGENCE_RATE",
        "FORCED_EXIT_RATE",
        "GROSS_PNL",
        "TRANSACTION_COST",
        "FINANCING_COST",
        "TOTAL_COST",
        "NET_PNL",
    ],
    "SSD trade statistics"
)

require_columns(
    eg_stats,
    [
        "PERIOD",
        "N_TRADES",
        "HIT_RATE",
        "HOLDING_DAYS_MEAN",
        "HOLDING_DAYS_MEDIAN",
        "CONVERGENCE_RATE",
        "FORCED_WINDOW_EXIT_RATE",
        "GROSS_PNL",
        "TRANSACTION_COST",
        "FINANCING_COST",
        "TOTAL_COST",
        "NET_PNL",
    ],
    "EG trade statistics"
)

require_columns(
    ssd_selection,
    [
        "BLOCK_TYPE",
        "PAIR_ID",
    ],
    "SSD selection"
)

require_columns(
    eg_selection,
    [
        "BLOCK_TYPE",
        "PAIR_ID",
    ],
    "EG selection"
)


# =============================================================================
# 2. AUDIT THAT FROZEN METHODS REALLY PASSED
# =============================================================================

if not ssd_audit[
    "STATUS"
].astype(str).str.upper().eq("PASS").all():
    raise RuntimeError(
        "SSD final audit contains a non-PASS row."
    )

if not eg_audit[
    "STATUS"
].astype(str).str.upper().eq("PASS").all():
    raise RuntimeError(
        "EG final audit contains a non-PASS row."
    )


ssd_full_1x = one_row(
    ssd_perf,
    COST_MULTIPLIER=1.0,
    PERIOD="FULL"
)

eg_full_1x = one_row(
    eg_perf,
    COST_MULTIPLIER=1.0,
    PERIOD="FULL"
)

ssd_dev_1x = one_row(
    ssd_perf,
    COST_MULTIPLIER=1.0,
    PERIOD="DEVELOPMENT"
)

eg_dev_1x = one_row(
    eg_perf,
    COST_MULTIPLIER=1.0,
    PERIOD="DEVELOPMENT"
)

ssd_oos_1x = one_row(
    ssd_perf,
    COST_MULTIPLIER=1.0,
    PERIOD="OOS"
)

eg_oos_1x = one_row(
    eg_perf,
    COST_MULTIPLIER=1.0,
    PERIOD="OOS"
)


# Mandatory like-for-like window check.
ssd_start = pd.Timestamp(
    ssd_full_1x["START_DATE"]
).normalize()

ssd_end = pd.Timestamp(
    ssd_full_1x["END_DATE"]
).normalize()

eg_start = pd.Timestamp(
    eg_full_1x["START_DATE"]
).normalize()

eg_end = pd.Timestamp(
    eg_full_1x["END_DATE"]
).normalize()

if (
    ssd_start != eg_start
    or
    ssd_end != eg_end
):
    raise RuntimeError(
        "SSD and EG do not use the same FULL comparison window.\n"
        f"SSD: {ssd_start.date()} to {ssd_end.date()}\n"
        f"EG:  {eg_start.date()} to {eg_end.date()}"
    )


# =============================================================================
# 3. MAIN SIDE-BY-SIDE COMPARISON
# =============================================================================

ssd_full_stats = one_row(
    ssd_stats,
    PERIOD="FULL"
)

eg_full_stats = one_row(
    eg_stats,
    PERIOD="FULL"
)

ssd_oos_stats = one_row(
    ssd_stats,
    PERIOD="OOS"
)

eg_oos_stats = one_row(
    eg_stats,
    PERIOD="OOS"
)

ssd_market_full = one_row(
    ssd_market,
    PERIOD="FULL"
)

eg_market_full = one_row(
    eg_market,
    PERIOD="FULL"
)

ssd_capacity_value = float(
    ssd_capacity[
        "STRATEGY_CAPACITY_RUPEES"
    ].iloc[0]
)

eg_capacity_value = float(
    eg_capacity[
        "STRATEGY_CAPACITY_RUPEES"
    ].iloc[0]
)

ssd_selected_full = len(
    ssd_selection
)

eg_selected_full = len(
    eg_selection
)

ssd_selected_oos = int(
    ssd_selection[
        "BLOCK_TYPE"
    ].astype(str).str.upper().eq("OOS").sum()
)

eg_selected_oos = int(
    eg_selection[
        "BLOCK_TYPE"
    ].astype(str).str.upper().eq("OOS").sum()
)


main_rows = [
    ["Comparison start date", str(ssd_start.date()), str(eg_start.date())],
    ["Comparison end date", str(ssd_end.date()), str(eg_end.date())],
    ["Selected pair-periods", ssd_selected_full, eg_selected_full],
    ["Trades", int(ssd_full_stats["N_TRADES"]), int(eg_full_stats["N_TRADES"])],
    ["Hit rate", pct(ssd_full_stats["HIT_RATE"]), pct(eg_full_stats["HIT_RATE"])],
    ["Convergence rate", pct(ssd_full_stats["CONVERGENCE_RATE"]), pct(eg_full_stats["CONVERGENCE_RATE"])],
    ["Forced-close rate", pct(ssd_full_stats["FORCED_EXIT_RATE"]), pct(eg_full_stats["FORCED_WINDOW_EXIT_RATE"])],
    ["Mean holding days", pct(ssd_full_stats["HOLDING_DAYS_MEAN"]), pct(eg_full_stats["HOLDING_DAYS_MEAN"])],
    ["Median holding days", pct(ssd_full_stats["HOLDING_DAYS_MEDIAN"]), pct(eg_full_stats["HOLDING_DAYS_MEDIAN"])],
    ["Gross return before costs", pct(one_row(ssd_perf, COST_MULTIPLIER=0.0, PERIOD="FULL")["TOTAL_RETURN"]), pct(one_row(eg_perf, COST_MULTIPLIER=0.0, PERIOD="FULL")["TOTAL_RETURN"])],
    ["Net return at 1x costs", pct(ssd_full_1x["TOTAL_RETURN"]), pct(eg_full_1x["TOTAL_RETURN"])],
    ["CAGR at 1x", pct(ssd_full_1x["CAGR"]), pct(eg_full_1x["CAGR"])],
    ["Annualized volatility", pct(ssd_full_1x["ANNUALIZED_VOLATILITY"]), pct(eg_full_1x["ANNUALIZED_VOLATILITY"])],
    ["Sharpe", pct(ssd_full_1x["SHARPE_RF0"]), pct(eg_full_1x["SHARPE_RF0"])],
    ["Sortino", pct(ssd_full_1x["SORTINO_RF0"]), pct(eg_full_1x["SORTINO_RF0"])],
    ["Maximum drawdown", pct(ssd_full_1x["MAX_DRAWDOWN"]), pct(eg_full_1x["MAX_DRAWDOWN"])],
    ["Calmar", pct(ssd_full_1x["CALMAR"]), pct(eg_full_1x["CALMAR"])],
    ["Average gross exposure / NAV", pct(ssd_full_1x["AVG_GROSS_EXPOSURE_TO_NAV"]), pct(eg_full_1x["AVG_GROSS_EXPOSURE_TO_NAV"])],
    ["Average absolute net exposure / NAV", pct(ssd_full_1x["AVG_ABS_NET_EXPOSURE_TO_NAV"]), pct(eg_full_1x["AVG_ABS_NET_EXPOSURE_TO_NAV"])],
    ["Total turnover / NAV", pct(ssd_full_1x["TOTAL_TURNOVER_TO_NAV"]), pct(eg_full_1x["TOTAL_TURNOVER_TO_NAV"])],
    ["NIFTY500 beta", pct(ssd_market_full["BETA_TO_NIFTY500"]), pct(eg_market_full["BETA_TO_NIFTY500"])],
    ["NIFTY500 correlation", pct(ssd_market_full["CORRELATION_TO_NIFTY500"]), pct(eg_market_full["CORRELATION_TO_NIFTY500"])],
    ["Capacity rupees", ssd_capacity_value, eg_capacity_value],
    ["Transaction costs / initial NAV", pct(ssd_full_stats["TRANSACTION_COST"]), pct(eg_full_stats["TRANSACTION_COST"])],
    ["Financing costs / initial NAV", pct(ssd_full_stats["FINANCING_COST"]), pct(eg_full_stats["FINANCING_COST"])],
    ["Total costs / initial NAV", pct(ssd_full_stats["TOTAL_COST"]), pct(eg_full_stats["TOTAL_COST"])],
    ["OOS selected pair-periods", ssd_selected_oos, eg_selected_oos],
    ["OOS trades", int(ssd_oos_stats["N_TRADES"]), int(eg_oos_stats["N_TRADES"])],
    ["OOS net return at 1x", pct(ssd_oos_1x["TOTAL_RETURN"]), pct(eg_oos_1x["TOTAL_RETURN"])],
]

main_comparison = pd.DataFrame(
    main_rows,
    columns=[
        "METRIC",
        "SSD",
        "ENGLE_GRANGER",
    ]
)

main_comparison.to_csv(
    OUTPUT_DIR
    / "01_MAIN_SSD_VS_EG_COMPARISON.csv",
    index=False
)


# =============================================================================
# 4. COST SENSITIVITY
# =============================================================================

cost_rows = []

for multiplier in [
    0.0,
    0.5,
    1.0,
    2.0,
]:
    s = one_row(
        ssd_perf,
        COST_MULTIPLIER=multiplier,
        PERIOD="FULL"
    )

    e = one_row(
        eg_perf,
        COST_MULTIPLIER=multiplier,
        PERIOD="FULL"
    )

    cost_rows.append(
        {
            "COST_MULTIPLIER":
                multiplier,

            "SSD_TOTAL_RETURN":
                float(
                    s[
                        "TOTAL_RETURN"
                    ]
                ),

            "ENGLE_GRANGER_TOTAL_RETURN":
                float(
                    e[
                        "TOTAL_RETURN"
                    ]
                ),

            "EG_MINUS_SSD_RETURN":
                float(
                    e[
                        "TOTAL_RETURN"
                    ]
                    -
                    s[
                        "TOTAL_RETURN"
                    ]
                ),
        }
    )


cost_comparison = pd.DataFrame(
    cost_rows
)

cost_comparison.to_csv(
    OUTPUT_DIR
    / "02_COST_SENSITIVITY_COMPARISON.csv",
    index=False
)


# =============================================================================
# 5. ROBUSTNESS COMPARISON
# =============================================================================

ssd_trade_count_col = (
    "N_COMPLETED_TRADES"
    if
    "N_COMPLETED_TRADES"
    in
    ssd_robust.columns
    else
    "N_TRADES"
)

eg_trade_count_col = (
    "N_TRADES"
    if
    "N_TRADES"
    in
    eg_robust.columns
    else
    "N_COMPLETED_TRADES"
)

ssd_r = ssd_robust[
    [
        "ROBUSTNESS_CATEGORY",
        "ROBUSTNESS_VARIANT",
        "TOTAL_RETURN",
        "N_SELECTED_PAIR_PERIODS",
        ssd_trade_count_col,
    ]
].copy()

ssd_r = ssd_r.rename(
    columns={
        "TOTAL_RETURN":
            "SSD_TOTAL_RETURN",

        "N_SELECTED_PAIR_PERIODS":
            "SSD_SELECTED_PAIR_PERIODS",

        ssd_trade_count_col:
            "SSD_TRADES",
    }
)

eg_r = eg_robust[
    [
        "ROBUSTNESS_CATEGORY",
        "ROBUSTNESS_VARIANT",
        "TOTAL_RETURN",
        "N_SELECTED_PAIR_PERIODS",
        eg_trade_count_col,
    ]
].copy()

eg_r = eg_r.rename(
    columns={
        "TOTAL_RETURN":
            "ENGLE_GRANGER_TOTAL_RETURN",

        "N_SELECTED_PAIR_PERIODS":
            "EG_SELECTED_PAIR_PERIODS",

        eg_trade_count_col:
            "EG_TRADES",
    }
)

robustness = ssd_r.merge(
    eg_r,
    on=[
        "ROBUSTNESS_CATEGORY",
        "ROBUSTNESS_VARIANT",
    ],
    how="outer"
)

robustness[
    "EG_MINUS_SSD_RETURN"
] = (
    robustness[
        "ENGLE_GRANGER_TOTAL_RETURN"
    ]
    -
    robustness[
        "SSD_TOTAL_RETURN"
    ]
)

robustness = robustness.sort_values(
    [
        "ROBUSTNESS_CATEGORY",
        "ROBUSTNESS_VARIANT",
    ]
).reset_index(
    drop=True
)

robustness.to_csv(
    OUTPUT_DIR
    / "03_ROBUSTNESS_COMPARISON.csv",
    index=False
)


# =============================================================================
# 6. DEVELOPMENT / OOS COMPARISON
# =============================================================================

period_rows = []

for period in [
    "FULL",
    "DEVELOPMENT",
    "OOS",
]:
    s = one_row(
        ssd_perf,
        COST_MULTIPLIER=1.0,
        PERIOD=period
    )

    e = one_row(
        eg_perf,
        COST_MULTIPLIER=1.0,
        PERIOD=period
    )

    period_rows.append(
        {
            "PERIOD":
                period,

            "SSD_TOTAL_RETURN":
                float(
                    s[
                        "TOTAL_RETURN"
                    ]
                ),

            "ENGLE_GRANGER_TOTAL_RETURN":
                float(
                    e[
                        "TOTAL_RETURN"
                    ]
                ),

            "SSD_CAGR":
                float(
                    s[
                        "CAGR"
                    ]
                ),

            "ENGLE_GRANGER_CAGR":
                float(
                    e[
                        "CAGR"
                    ]
                ),

            "SSD_SHARPE":
                float(
                    s[
                        "SHARPE_RF0"
                    ]
                )
                if
                pd.notna(
                    s[
                        "SHARPE_RF0"
                    ]
                )
                else
                np.nan,

            "ENGLE_GRANGER_SHARPE":
                float(
                    e[
                        "SHARPE_RF0"
                    ]
                )
                if
                pd.notna(
                    e[
                        "SHARPE_RF0"
                    ]
                )
                else
                np.nan,

            "SSD_MAX_DRAWDOWN":
                float(
                    s[
                        "MAX_DRAWDOWN"
                    ]
                ),

            "ENGLE_GRANGER_MAX_DRAWDOWN":
                float(
                    e[
                        "MAX_DRAWDOWN"
                    ]
                ),
        }
    )


period_comparison = pd.DataFrame(
    period_rows
)

period_comparison.to_csv(
    OUTPUT_DIR
    / "04_FULL_DEV_OOS_COMPARISON.csv",
    index=False
)


# =============================================================================
# 7. SHORT INTERPRETATION TABLE
# =============================================================================

ssd_net = float(
    ssd_full_1x[
        "TOTAL_RETURN"
    ]
)

eg_net = float(
    eg_full_1x[
        "TOTAL_RETURN"
    ]
)

ssd_conv = float(
    ssd_full_stats[
        "CONVERGENCE_RATE"
    ]
)

eg_conv = float(
    eg_full_stats[
        "CONVERGENCE_RATE"
    ]
)

ssd_dd = float(
    ssd_full_1x[
        "MAX_DRAWDOWN"
    ]
)

eg_dd = float(
    eg_full_1x[
        "MAX_DRAWDOWN"
    ]
)


interpretation = pd.DataFrame(
    [
        {
            "QUESTION":
                "Which method had the better full-period net result?",

            "ANSWER":
                (
                    "ENGLE_GRANGER"
                    if
                    eg_net
                    >
                    ssd_net
                    else
                    "SSD"
                ),

            "EVIDENCE":
                f"SSD={ssd_net:.4%}; EG={eg_net:.4%}",
        },
        {
            "QUESTION":
                "Which method had more trades?",

            "ANSWER":
                (
                    "SSD"
                    if
                    int(
                        ssd_full_stats[
                            "N_TRADES"
                        ]
                    )
                    >
                    int(
                        eg_full_stats[
                            "N_TRADES"
                        ]
                    )
                    else
                    "ENGLE_GRANGER"
                ),

            "EVIDENCE":
                (
                    f"SSD={int(ssd_full_stats['N_TRADES'])}; "
                    f"EG={int(eg_full_stats['N_TRADES'])}"
                ),
        },
        {
            "QUESTION":
                "Which method had the higher convergence rate?",

            "ANSWER":
                (
                    "ENGLE_GRANGER"
                    if
                    eg_conv
                    >
                    ssd_conv
                    else
                    "SSD"
                ),

            "EVIDENCE":
                f"SSD={ssd_conv:.1%}; EG={eg_conv:.1%}",
        },
        {
            "QUESTION":
                "Which method had the smaller maximum drawdown?",

            "ANSWER":
                (
                    "ENGLE_GRANGER"
                    if
                    abs(
                        eg_dd
                    )
                    <
                    abs(
                        ssd_dd
                    )
                    else
                    "SSD"
                ),

            "EVIDENCE":
                f"SSD={ssd_dd:.2%}; EG={eg_dd:.2%}",
        },
        {
            "QUESTION":
                "What does the EG OOS result mean?",

            "ANSWER":
                "NO_OOS_TRADING_EVIDENCE",

            "EVIDENCE":
                (
                    f"EG selected {eg_selected_oos} OOS pair-periods "
                    f"and made {int(eg_oos_stats['N_TRADES'])} OOS trades."
                ),
        },
        {
            "QUESTION":
                "What does the SSD OOS result mean?",

            "ANSWER":
                "ESSENTIALLY_FLAT_PSEUDO_OOS",

            "EVIDENCE":
                (
                    f"SSD OOS net return={float(ssd_oos_1x['TOTAL_RETURN']):.4%}; "
                    f"trades={int(ssd_oos_stats['N_TRADES'])}."
                ),
        },
    ]
)

interpretation.to_csv(
    OUTPUT_DIR
    / "05_INTERPRETATION_CHECKPOINTS.csv",
    index=False
)


# =============================================================================
# 8. FINAL COMPARISON AUDIT
# =============================================================================

audit_rows = [
    [
        "SSD_FINAL_AUDIT_ALL_PASS",
        True,
    ],
    [
        "EG_FINAL_AUDIT_ALL_PASS",
        True,
    ],
    [
        "IDENTICAL_FULL_START_DATE",
        ssd_start
        ==
        eg_start,
    ],
    [
        "IDENTICAL_FULL_END_DATE",
        ssd_end
        ==
        eg_end,
    ],
    [
        "IDENTICAL_COST_MULTIPLIERS",
        set(
            pd.to_numeric(
                ssd_perf[
                    "COST_MULTIPLIER"
                ],
                errors="coerce"
            ).dropna().unique()
        )
        ==
        set(
            pd.to_numeric(
                eg_perf[
                    "COST_MULTIPLIER"
                ],
                errors="coerce"
            ).dropna().unique()
        ),
    ],
    [
        "MAIN_COMPARISON_CREATED",
        not main_comparison.empty,
    ],
    [
        "COST_COMPARISON_CREATED",
        len(
            cost_comparison
        )
        ==
        4,
    ],
    [
        "ROBUSTNESS_COMPARISON_CREATED",
        not robustness.empty,
    ],
]

comparison_audit = pd.DataFrame(
    audit_rows,
    columns=[
        "CHECK",
        "PASS",
    ]
)

comparison_audit[
    "STATUS"
] = np.where(
    comparison_audit[
        "PASS"
    ],
    "PASS",
    "FAIL"
)

comparison_audit.to_csv(
    OUTPUT_DIR
    / "06_COMPARISON_AUDIT.csv",
    index=False
)

if not comparison_audit[
    "PASS"
].all():
    raise RuntimeError(
        "Final SSD-vs-EG comparison audit failed:\n\n"
        +
        comparison_audit[
            ~comparison_audit[
                "PASS"
            ]
        ].to_string(
            index=False
        )
    )


# =============================================================================
# 9. CONSOLE SUMMARY
# =============================================================================

print(
    "="
    *
    90
)

print(
    "FINAL SSD vs ENGLE-GRANGER COMPARISON — PASS"
)

print(
    "="
    *
    90
)

print(
    f"Common window: {ssd_start.date()} to {ssd_end.date()}"
)

print(
    f"SSD 1x net return: {ssd_net:.2%}"
)

print(
    f"EG  1x net return: {eg_net:.2%}"
)

print(
    f"SSD trades: {int(ssd_full_stats['N_TRADES'])}"
)

print(
    f"EG trades:  {int(eg_full_stats['N_TRADES'])}"
)

print(
    f"SSD convergence: {ssd_conv:.1%}"
)

print(
    f"EG convergence:  {eg_conv:.1%}"
)

print(
    f"SSD max drawdown: {ssd_dd:.2%}"
)

print(
    f"EG max drawdown:  {eg_dd:.2%}"
)

print(
    "\nOutput folder:"
)

print(
    OUTPUT_DIR
)

print(
    "\nCreated:"
)

for name in [
    "01_MAIN_SSD_VS_EG_COMPARISON.csv",
    "02_COST_SENSITIVITY_COMPARISON.csv",
    "03_ROBUSTNESS_COMPARISON.csv",
    "04_FULL_DEV_OOS_COMPARISON.csv",
    "05_INTERPRETATION_CHECKPOINTS.csv",
    "06_COMPARISON_AUDIT.csv",
]:
    print(
        OUTPUT_DIR
        /
        name
    )
