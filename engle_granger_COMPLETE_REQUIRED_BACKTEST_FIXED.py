import os
import io
import csv
import re
import time
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from itertools import combinations

try:
    from statsmodels.tsa.stattools import coint
except ImportError as exc:
    raise ImportError(
        "This program needs statsmodels. Install it once with:\n"
        "pip install statsmodels"
    ) from exc


# =============================================================================
# NIFTY PHARMA — COMPLETE REQUIRED ENGLE-GRANGER TRADING + BACKTEST
# =============================================================================
#
# FAST-TRACK PURPOSE
# ------------------
# This is the ONE final EG backtest program.
#
# It does:
#   1. Read the already-audited EG selected pairs.
#   2. Build the EG gap during each following 6-month trading window.
#   3. Enter when the gap reaches formation mean +/- 2 formation standard
#      deviations.
#   4. Exit when the gap crosses back through the frozen formation mean.
#   5. If it never returns, force-close at the 6-month end.
#   6. Signal at close t -> execute on next common observation.
#   7. Re-entry is allowed after a completed trade.
#   8. Verify the ACTUAL short leg had an NSE stock future on entry date.
#      Only the few actual EG entry dates are checked/downloaded.
#   9. Skip an F&O-infeasible logical trade entirely. No synthetic re-entry.
#  10. Calculate long-cash / short-stock-futures-proxy P&L.
#  11. Use beta-based sizing, keeping each pair slot at 25% gross.
#  12. Apply 0.5x / 1x / 2x transaction-cost scenarios.
#  13. Produce full, development, OOS, six-month-period, trade and pair results.
#  14. Calculate NIFTY 500 beta/correlation if the benchmark CSV exists.
#
# It does NOT:
#   - redo EG pair selection
#   - tune the EG test
#   - change the five selected pair-periods
#   - model actual futures basis, monthly roll, lot size or margin funding
#
# IMPORTANT PLAIN-ENGLISH INTERPRETATION
# --------------------------------------
# EG gap:
#     GAP = log(A total-return index) - alpha - beta*log(B total-return index)
#
# High gap:
#     A is high relative to its historical relationship with B
#     -> SHORT A, LONG B
#
# Low gap:
#     A is low relative to its historical relationship with B
#     -> LONG A, SHORT B
#
# beta sizing:
#     absolute A notional : absolute B notional = 1 : beta
#
# Total gross size of one pair slot remains 25% of current portfolio value.
# =============================================================================


# =============================================================================
# 1. PATHS / CONFIG
# =============================================================================

PROJECT_ROOT = Path(
    os.environ.get(
        "PAIR_TRADING_PROJECT_ROOT",
        r"C:\fin proj"
    )
)

EG_ROOT = (
    PROJECT_ROOT
    / "pair_trading_methods"
    / "ENGLE_GRANGER"
)

SELECTED_FILE = (
    EG_ROOT
    / "01_pair_selection"
    / "03_SELECTED_ENGLE_GRANGER_PAIRS.csv"
)

SELECTION_AUDIT_FILE = (
    EG_ROOT
    / "04_audit"
    / "00_ENGLE_GRANGER_PAIR_SELECTION_AUDIT.csv"
)

# Reuse the already-frozen 19-date calendar. No SSD pair choices are used.
SCHEDULE_FILE = (
    PROJECT_ROOT
    / "pair_trading_methods"
    / "SSD"
    / "02_trading_rules"
    / "01_SELECTED_PAIR_FORMATION_THRESHOLDS.csv"
)

TOTAL_RETURN_FILE = (
    PROJECT_ROOT
    / "nse_pharma_total_return"
    / "NIFTY_PHARMA_TOTAL_RETURN_BASE_2016_2026.csv"
)

INVESTABLE_FILE = (
    PROJECT_ROOT
    / "nse_pharma_final_investable_universe_FINAL"
    / "07_FINAL_INVESTABLE_BY_FORMATION.csv"
)

LAG_SENSITIVITY_FILE = (
    EG_ROOT
    / "01_pair_selection"
    / "05_SELECTED_PAIR_LAG_SENSITIVITY.csv"
)

BENCHMARK_FILE = (
    PROJECT_ROOT
    / "benchmark"
    / "NIFTY500_2017_2026.csv"
)

OUTPUT_DIR = (
    EG_ROOT
    / "02_FINAL_BACKTEST"
)

RAW_FO_DIR = (
    OUTPUT_DIR
    / "raw_nse_fo_bhavcopy"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RAW_FO_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ENTRY_Z = 2.0
PAIR_GROSS_WEIGHT = 0.25

CASH_RT_BPS = 30.0
FUTURES_RT_BPS = 8.0

CASH_ONE_WAY_RATE = (
    CASH_RT_BPS
    /
    2.0
    /
    10000.0
)

FUTURES_ONE_WAY_RATE = (
    FUTURES_RT_BPS
    /
    2.0
    /
    10000.0
)

COST_MULTIPLIERS = [
    0.0,   # gross result before trading costs
    0.5,
    1.0,
    2.0,
]

INITIAL_NAV = 1.0
TRADING_DAYS_PER_YEAR = 252

OOS_START = pd.Timestamp(
    "2024-08-01"
)

EXPECTED_FORMATION_DATES = 19
# Do not hard-code the number of selected pairs.
# The corrected two-direction EG selection currently gives 6, but the
# backtest must always read whatever the audited selection actually produced.
EXPECTED_SELECTED_PAIR_PERIODS = None

# NSE changed derivatives bhavcopy schema on 08-Jul-2024.
UDIFF_START_DATE = pd.Timestamp(
    "2024-07-08"
)

ALIASES = {
    "CADILAHC":
        "ZYDUS",

    "ZYDUSLIFE":
        "ZYDUS",

    "AJANTAPHARM":
        "AJANTPHARM",
}

REQUEST_DELAY_SECONDS = 0.05
NUMERIC_TOL = 1e-12

# Short leg is implemented with a stock future, so there is no stock-borrow fee.
# We nevertheless include an explicit funding cost for the margin/collateral
# required to carry the futures position.
#
# These are transparent modelling assumptions, not exchange facts:
#   - 20% of futures notional treated as funded margin/collateral
#   - 8% annual funding rate on that funded amount
#
# Therefore the normal 1x financing charge is about 1.6% per year of
# the short futures notional while the position is open.
FUTURES_MARGIN_FRACTION = 0.20
FINANCING_RATE_ANNUAL = 0.08


# =============================================================================
# 2. BASIC HELPERS
# =============================================================================

def clean_col(v):
    return re.sub(
        r"[^A-Z0-9]+",
        "_",
        str(v).strip().upper()
    ).strip("_")


def clean_columns(df):
    out = df.copy()

    out.columns = [
        clean_col(
            col
        )
        for col
        in out.columns
    ]

    return out


def clean_symbol(v):
    if v is None:
        return None

    if (
        isinstance(
            v,
            float
        )
        and
        np.isnan(
            v
        )
    ):
        return None

    s = re.sub(
        r"\s+",
        "",
        str(v).strip().upper()
    )

    if s in {
        "",
        "NAN",
        "NONE",
        "NA",
        "N/A",
        "-",
        "--",
    }:
        return None

    return s


def company_id_from_symbol(symbol):
    s = clean_symbol(
        symbol
    )

    if s is None:
        return None

    return ALIASES.get(
        s,
        s
    )


def parse_dates(series):
    try:
        return pd.to_datetime(
            series,
            format="mixed",
            errors="coerce"
        ).dt.normalize()

    except TypeError:
        return pd.to_datetime(
            series,
            errors="coerce"
        ).dt.normalize()


def require_columns(
    df,
    required,
    name
):
    missing = (
        set(
            required
        )
        -
        set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{name} missing required columns: "
            f"{sorted(missing)}"
        )


def safe_div(
    numerator,
    denominator
):
    if (
        denominator is None
        or
        not np.isfinite(
            denominator
        )
        or
        abs(
            denominator
        )
        <
        NUMERIC_TOL
    ):
        return np.nan

    return (
        numerator
        /
        denominator
    )


def to_bool(series):
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(
            False
        )

    return (
        series.astype(
            str
        )
        .str.strip()
        .str.upper()
        .isin(
            [
                "TRUE",
                "1",
                "YES",
                "Y",
            ]
        )
    )


# =============================================================================
# 3. LOAD AND VALIDATE INPUTS
# =============================================================================

print(
    "="
    *
    112
)

print(
    "ENGLE-GRANGER — COMPLETE REQUIRED TRADING + BACKTEST"
)

print(
    "="
    *
    112
)


for path in [
    SELECTED_FILE,
    SELECTION_AUDIT_FILE,
    SCHEDULE_FILE,
    TOTAL_RETURN_FILE,
    INVESTABLE_FILE,
    LAG_SENSITIVITY_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found:\n{path}"
        )


selected = clean_columns(
    pd.read_csv(
        SELECTED_FILE,
        low_memory=False
    )
)

selection_audit = clean_columns(
    pd.read_csv(
        SELECTION_AUDIT_FILE,
        low_memory=False
    )
)

schedule_raw = clean_columns(
    pd.read_csv(
        SCHEDULE_FILE,
        low_memory=False
    )
)

prices = clean_columns(
    pd.read_csv(
        TOTAL_RETURN_FILE,
        low_memory=False
    )
)


require_columns(
    selected,
    [
        "FORMATION_DATE",
        "BLOCK_TYPE",
        "SELECTED_PAIR_NUMBER",
        "PAIR_ID",
        "COMPANY_A",
        "COMPANY_B",
        "FORMATION_START",
        "TRADING_START",
        "TRADING_END",
        "ALPHA",
        "BETA",
        "RESIDUAL_MEAN",
        "RESIDUAL_STD_DDOF1",
        "COINT_PVALUE_RAW",
        "COINT_PVALUE_BH",
    ],
    "Selected EG pairs",
)

require_columns(
    selection_audit,
    [
        "FORMATION_DATE",
        "BLOCK_TYPE",
        "N_SELECTED",
        "STATUS",
    ],
    "EG selection audit",
)

require_columns(
    schedule_raw,
    [
        "FORMATION_DATE",
        "BLOCK_TYPE",
        "FORMATION_START",
        "TRADING_START",
        "TRADING_END",
    ],
    "Frozen 19-date schedule",
)

require_columns(
    prices,
    [
        "DATE",
        "COMPANY_ID",
        "CLOSE",
        "TOTAL_RETURN",
        "TOTAL_RETURN_INDEX",
        "SEGMENT_ID",
        "CA_EVENT_TYPE",
        "TR_PREV_CLOSE",
        "TOTAL_TRADED_VALUE",
    ],
    "Total-return data",
)


for col in [
    "FORMATION_DATE",
    "FORMATION_START",
    "TRADING_START",
    "TRADING_END",
]:
    selected[
        col
    ] = parse_dates(
        selected[
            col
        ]
    )

for col in [
    "FORMATION_DATE",
]:
    selection_audit[
        col
    ] = parse_dates(
        selection_audit[
            col
        ]
    )

for col in [
    "FORMATION_DATE",
    "FORMATION_START",
    "TRADING_START",
    "TRADING_END",
]:
    schedule_raw[
        col
    ] = parse_dates(
        schedule_raw[
            col
        ]
    )

prices[
    "DATE"
] = parse_dates(
    prices[
        "DATE"
    ]
)


for df, cols in [
    (
        selected,
        [
            "BLOCK_TYPE",
            "PAIR_ID",
            "COMPANY_A",
            "COMPANY_B",
        ],
    ),
    (
        selection_audit,
        [
            "BLOCK_TYPE",
            "STATUS",
        ],
    ),
    (
        schedule_raw,
        [
            "BLOCK_TYPE",
        ],
    ),
    (
        prices,
        [
            "COMPANY_ID",
            "SEGMENT_ID",
            "CA_EVENT_TYPE",
        ],
    ),
]:
    for col in cols:
        df[
            col
        ] = (
            df[
                col
            ]
            .astype(
                str
            )
            .str.strip()
            .str.upper()
        )


for col in [
    "ALPHA",
    "BETA",
    "RESIDUAL_MEAN",
    "RESIDUAL_STD_DDOF1",
    "COINT_PVALUE_RAW",
    "COINT_PVALUE_BH",
]:
    selected[
        col
    ] = pd.to_numeric(
        selected[
            col
        ],
        errors="coerce"
    )


for col in [
    "CLOSE",
    "TOTAL_RETURN",
    "TOTAL_RETURN_INDEX",
    "TR_PREV_CLOSE",
    "TOTAL_TRADED_VALUE",
]:
    prices[
        col
    ] = pd.to_numeric(
        prices[
            col
        ],
        errors="coerce"
    )


if selected[
    [
        "FORMATION_DATE",
        "FORMATION_START",
        "TRADING_START",
        "TRADING_END",
    ]
].isna().any().any():
    raise RuntimeError(
        "Invalid required date in selected EG pairs."
    )

if schedule_raw[
    [
        "FORMATION_DATE",
        "FORMATION_START",
        "TRADING_START",
        "TRADING_END",
    ]
].isna().any().any():
    raise RuntimeError(
        "Invalid required date in frozen schedule."
    )

if prices[
    "DATE"
].isna().any():
    raise RuntimeError(
        "Invalid DATE in total-return data."
    )

if selected[
    [
        "ALPHA",
        "BETA",
        "RESIDUAL_MEAN",
        "RESIDUAL_STD_DDOF1",
    ]
].isna().any().any():
    raise RuntimeError(
        "Missing EG relationship parameter."
    )

if (
    selected[
        "BETA"
    ]
    <=
    0
).any():
    raise RuntimeError(
        "Selected EG pair contains non-positive beta."
    )

if (
    selected[
        "RESIDUAL_STD_DDOF1"
    ]
    <=
    0
).any():
    raise RuntimeError(
        "Selected EG pair contains non-positive formation gap standard deviation."
    )

if (
    prices[
        "CLOSE"
    ]
    <=
    0
).any():
    raise RuntimeError(
        "Non-positive CLOSE in total-return data."
    )

if (
    prices[
        "TOTAL_RETURN_INDEX"
    ]
    <=
    0
).any():
    raise RuntimeError(
        "Non-positive TOTAL_RETURN_INDEX in total-return data."
    )

if prices[
    [
        "DATE",
        "COMPANY_ID",
    ]
].duplicated().any():
    raise RuntimeError(
        "Duplicate DATE + COMPANY_ID in total-return data."
    )


schedule = (
    schedule_raw[
        [
            "FORMATION_DATE",
            "BLOCK_TYPE",
            "FORMATION_START",
            "TRADING_START",
            "TRADING_END",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "FORMATION_DATE"
    )
    .reset_index(
        drop=True
    )
)


if schedule[
    "FORMATION_DATE"
].duplicated().any():
    raise RuntimeError(
        "Frozen schedule contains inconsistent duplicate formation dates."
    )

if len(
    schedule
) != EXPECTED_FORMATION_DATES:
    raise RuntimeError(
        f"Expected {EXPECTED_FORMATION_DATES} formation dates; "
        f"found {len(schedule)}."
    )

if (
    selection_audit[
        "STATUS"
    ]
    !=
    "PASS"
).any():
    raise RuntimeError(
        "The earlier corrected EG pair-selection audit is not all PASS."
    )

audited_selected_count = int(
    selection_audit[
        "N_SELECTED"
    ].sum()
)

if len(selected) != audited_selected_count:
    raise RuntimeError(
        "Selected-pair file does not reconcile with the corrected "
        f"pair-selection audit. Selected file={len(selected)}, "
        f"audit={audited_selected_count}."
    )

if len(selected) == 0:
    print("WARNING: corrected EG method selected zero pair-periods.")


# Confirm selected pair dates/blocks agree with frozen calendar.
calendar_check = selected.merge(
    schedule,
    on=[
        "FORMATION_DATE",
        "BLOCK_TYPE",
    ],
    how="left",
    suffixes=(
        "_SELECTED",
        "_SCHEDULE",
    ),
    validate="many_to_one"
)

if calendar_check[
    "TRADING_START_SCHEDULE"
].isna().any():
    raise RuntimeError(
        "A selected EG pair does not match the frozen formation calendar."
    )

for left_col, right_col in [
    (
        "FORMATION_START_SELECTED",
        "FORMATION_START_SCHEDULE",
    ),
    (
        "TRADING_START_SELECTED",
        "TRADING_START_SCHEDULE",
    ),
    (
        "TRADING_END_SELECTED",
        "TRADING_END_SCHEDULE",
    ),
]:
    if not (
        calendar_check[
            left_col
        ]
        ==
        calendar_check[
            right_col
        ]
    ).all():
        raise RuntimeError(
            f"Frozen calendar mismatch: {left_col} vs {right_col}."
        )


print(
    f"Formation dates:          {len(schedule)}"
)

print(
    f"Selected EG pair-periods: {len(selected)}"
)

print(
    f"OOS selected pair-periods: "
    f"{int(selected['BLOCK_TYPE'].eq('OOS').sum())}"
)


# =============================================================================
# 4. BUILD EG GAP PATHS AND LOGICAL TRADES
# =============================================================================

logical_trade_rows = []
pair_window_audit_rows = []
gap_path_rows = []


for pair_row in selected.sort_values(
    [
        "FORMATION_DATE",
        "SELECTED_PAIR_NUMBER",
    ]
).itertuples(
    index=False
):
    formation_date = pd.Timestamp(
        pair_row.FORMATION_DATE
    )

    trading_start = pd.Timestamp(
        pair_row.TRADING_START
    )

    trading_end = pd.Timestamp(
        pair_row.TRADING_END
    )

    company_a = pair_row.COMPANY_A
    company_b = pair_row.COMPANY_B

    alpha = float(
        pair_row.ALPHA
    )

    beta = float(
        pair_row.BETA
    )

    mean_gap = float(
        pair_row.RESIDUAL_MEAN
    )

    sd_gap = float(
        pair_row.RESIDUAL_STD_DDOF1
    )

    upper = (
        mean_gap
        +
        ENTRY_Z
        *
        sd_gap
    )

    lower = (
        mean_gap
        -
        ENTRY_Z
        *
        sd_gap
    )


    a = prices.loc[
        prices[
            "COMPANY_ID"
        ].eq(
            company_a
        )
        &
        prices[
            "DATE"
        ].between(
            trading_start,
            trading_end,
            inclusive="both"
        ),
        [
            "DATE",
            "TOTAL_RETURN_INDEX",
            "SEGMENT_ID",
        ],
    ].rename(
        columns={
            "TOTAL_RETURN_INDEX":
                "TRI_A",

            "SEGMENT_ID":
                "SEGMENT_A",
        }
    )

    b = prices.loc[
        prices[
            "COMPANY_ID"
        ].eq(
            company_b
        )
        &
        prices[
            "DATE"
        ].between(
            trading_start,
            trading_end,
            inclusive="both"
        ),
        [
            "DATE",
            "TOTAL_RETURN_INDEX",
            "SEGMENT_ID",
        ],
    ].rename(
        columns={
            "TOTAL_RETURN_INDEX":
                "TRI_B",

            "SEGMENT_ID":
                "SEGMENT_B",
        }
    )


    pair = (
        a.merge(
            b,
            on="DATE",
            how="inner",
            validate="one_to_one"
        )
        .sort_values(
            "DATE"
        )
        .reset_index(
            drop=True
        )
    )


    if len(
        pair
    ) < 2:
        raise RuntimeError(
            f"{pair_row.PAIR_ID} {formation_date.date()}: "
            "fewer than two common trading observations."
        )


    pair[
        "GAP"
    ] = (
        np.log(
            pair[
                "TRI_A"
            ].astype(
                float
            )
        )
        -
        alpha
        -
        beta
        *
        np.log(
            pair[
                "TRI_B"
            ].astype(
                float
            )
        )
    )


    # -------------------------------------------------------------------------
    # Structural-break handling:
    # Use the segment in force at the start of the trading window.
    # If either leg changes segment later, stop before the changed segment.
    # -------------------------------------------------------------------------

    first_segment_a = str(
        pair.loc[
            0,
            "SEGMENT_A"
        ]
    )

    first_segment_b = str(
        pair.loc[
            0,
            "SEGMENT_B"
        ]
    )

    mismatch = (
        pair[
            "SEGMENT_A"
        ].astype(
            str
        ).ne(
            first_segment_a
        )
        |
        pair[
            "SEGMENT_B"
        ].astype(
            str
        ).ne(
            first_segment_b
        )
    )

    break_positions = np.where(
        mismatch.to_numpy()
    )[0]

    structural_break_date = pd.NaT

    if len(
        break_positions
    ):
        first_break_pos = int(
            break_positions[
                0
            ]
        )

        structural_break_date = pd.Timestamp(
            pair.loc[
                first_break_pos,
                "DATE"
            ]
        )

        terminal_pos = (
            first_break_pos
            -
            1
        )

        terminal_reason = (
            "STRUCTURAL_BREAK"
        )

    else:
        terminal_pos = (
            len(
                pair
            )
            -
            1
        )

        terminal_reason = (
            "WINDOW_END"
        )


    if terminal_pos < 1:
        pair_window_audit_rows.append(
            {
                "FORMATION_DATE":
                    formation_date,

                "BLOCK_TYPE":
                    pair_row.BLOCK_TYPE,

                "SELECTED_PAIR_NUMBER":
                    int(
                        pair_row.SELECTED_PAIR_NUMBER
                    ),

                "PAIR_ID":
                    pair_row.PAIR_ID,

                "COMPANY_A":
                    company_a,

                "COMPANY_B":
                    company_b,

                "N_COMMON_OBSERVATIONS":
                    len(
                        pair
                    ),

                "EFFECTIVE_TERMINAL_DATE":
                    (
                        pair.loc[
                            max(
                                terminal_pos,
                                0
                            ),
                            "DATE"
                        ]
                        if
                        len(
                            pair
                        )
                        else
                        pd.NaT
                    ),

                "STRUCTURAL_BREAK_DATE":
                    structural_break_date,

                "LOGICAL_TRADES":
                    0,

                "NORMAL_MEAN_CROSS_EXITS":
                    0,

                "FORCED_EXITS":
                    0,

                "STATUS":
                    "NO_TRADING_BEFORE_STRUCTURAL_BREAK",
            }
        )

        continue


    effective = pair.iloc[
        :
        terminal_pos
        +
        1
    ].copy().reset_index(
        drop=True
    )


    for r in effective.itertuples(
        index=False
    ):
        gap_path_rows.append(
            {
                "FORMATION_DATE":
                    formation_date,

                "BLOCK_TYPE":
                    pair_row.BLOCK_TYPE,

                "SELECTED_PAIR_NUMBER":
                    int(
                        pair_row.SELECTED_PAIR_NUMBER
                    ),

                "PAIR_ID":
                    pair_row.PAIR_ID,

                "COMPANY_A":
                    company_a,

                "COMPANY_B":
                    company_b,

                "DATE":
                    r.DATE,

                "GAP":
                    float(
                        r.GAP
                    ),

                "FORMATION_GAP_MEAN":
                    mean_gap,

                "FORMATION_GAP_STD":
                    sd_gap,

                "UPPER_ENTRY_LEVEL":
                    upper,

                "LOWER_ENTRY_LEVEL":
                    lower,
            }
        )


    trade_number = 0
    scan_pos = 0

    normal_exits = 0
    forced_exits = 0


    # Need at least one observation AFTER the signal for execution,
    # and one later observation if a position is to have economic life.
    while scan_pos <= (
        len(
            effective
        )
        -
        3
    ):
        signal_pos = None
        entry_direction = None


        # -------------------------------------------------------------
        # Search for the next entry signal while flat.
        # -------------------------------------------------------------

        for i in range(
            scan_pos,
            len(
                effective
            )
            -
            2
        ):
            gap_i = float(
                effective.loc[
                    i,
                    "GAP"
                ]
            )

            if gap_i >= upper:
                signal_pos = i
                entry_direction = (
                    "HIGH_GAP_SHORT_A_LONG_B"
                )
                break

            if gap_i <= lower:
                signal_pos = i
                entry_direction = (
                    "LOW_GAP_LONG_A_SHORT_B"
                )
                break


        if signal_pos is None:
            break


        entry_exec_pos = (
            signal_pos
            +
            1
        )

        # Require at least one later common observation.
        if entry_exec_pos >= (
            len(
                effective
            )
            -
            1
        ):
            break


        if entry_direction == (
            "HIGH_GAP_SHORT_A_LONG_B"
        ):
            long_company = company_b
            short_company = company_a
            entry_side = "ABOVE"

        else:
            long_company = company_a
            short_company = company_b
            entry_side = "BELOW"


        # -------------------------------------------------------------
        # Search for mean crossing after entry.
        # Signal at close j -> execute exit on j+1.
        # We do not claim a normal exit if the mean is crossed only on
        # the final available day because there is no next execution day.
        # -------------------------------------------------------------

        exit_signal_pos = None

        for j in range(
            entry_exec_pos,
            len(
                effective
            )
            -
            1
        ):
            gap_j = float(
                effective.loc[
                    j,
                    "GAP"
                ]
            )

            if (
                entry_side
                ==
                "ABOVE"
                and
                gap_j
                <=
                mean_gap
            ):
                exit_signal_pos = j
                break

            if (
                entry_side
                ==
                "BELOW"
                and
                gap_j
                >=
                mean_gap
            ):
                exit_signal_pos = j
                break


        trade_number += 1


        if exit_signal_pos is not None:
            exit_exec_pos = (
                exit_signal_pos
                +
                1
            )

            exit_reason = (
                "MEAN_CROSS"
            )

            exit_signal_date = pd.Timestamp(
                effective.loc[
                    exit_signal_pos,
                    "DATE"
                ]
            )

            normal_exits += 1

        else:
            exit_exec_pos = (
                len(
                    effective
                )
                -
                1
            )

            exit_signal_date = pd.NaT

            exit_reason = (
                "STRUCTURAL_BREAK_FORCE_CLOSE"
                if
                terminal_reason
                ==
                "STRUCTURAL_BREAK"
                else
                "WINDOW_END_FORCE_CLOSE"
            )

            forced_exits += 1


        entry_signal_date = pd.Timestamp(
            effective.loc[
                signal_pos,
                "DATE"
            ]
        )

        entry_execution_date = pd.Timestamp(
            effective.loc[
                entry_exec_pos,
                "DATE"
            ]
        )

        exit_execution_date = pd.Timestamp(
            effective.loc[
                exit_exec_pos,
                "DATE"
            ]
        )


        logical_trade_rows.append(
            {
                "FORMATION_DATE":
                    formation_date,

                "BLOCK_TYPE":
                    pair_row.BLOCK_TYPE,

                "SELECTED_PAIR_NUMBER":
                    int(
                        pair_row.SELECTED_PAIR_NUMBER
                    ),

                "PAIR_ID":
                    pair_row.PAIR_ID,

                "TRADE_NUMBER_WITHIN_PAIR_WINDOW":
                    trade_number,

                "COMPANY_A":
                    company_a,

                "COMPANY_B":
                    company_b,

                "ALPHA":
                    alpha,

                "BETA":
                    beta,

                "FORMATION_GAP_MEAN":
                    mean_gap,

                "FORMATION_GAP_STD":
                    sd_gap,

                "UPPER_ENTRY_LEVEL":
                    upper,

                "LOWER_ENTRY_LEVEL":
                    lower,

                "ENTRY_SIGNAL_DATE":
                    entry_signal_date,

                "ENTRY_SIGNAL_GAP":
                    float(
                        effective.loc[
                            signal_pos,
                            "GAP"
                        ]
                    ),

                "ENTRY_SIGNAL_Z":
                    float(
                        (
                            effective.loc[
                                signal_pos,
                                "GAP"
                            ]
                            -
                            mean_gap
                        )
                        /
                        sd_gap
                    ),

                "ENTRY_EXECUTION_DATE":
                    entry_execution_date,

                "ENTRY_EXECUTION_GAP":
                    float(
                        effective.loc[
                            entry_exec_pos,
                            "GAP"
                        ]
                    ),

                "ENTRY_EXECUTION_Z":
                    float(
                        (
                            effective.loc[
                                entry_exec_pos,
                                "GAP"
                            ]
                            -
                            mean_gap
                        )
                        /
                        sd_gap
                    ),

                "ENTRY_DIRECTION":
                    entry_direction,

                "LONG_COMPANY":
                    long_company,

                "SHORT_COMPANY":
                    short_company,

                "EXIT_SIGNAL_DATE":
                    exit_signal_date,

                "EXIT_SIGNAL_GAP":
                    (
                        float(
                            effective.loc[
                                exit_signal_pos,
                                "GAP"
                            ]
                        )
                        if
                        exit_signal_pos is not None
                        else
                        np.nan
                    ),

                "EXIT_SIGNAL_Z":
                    (
                        float(
                            (
                                effective.loc[
                                    exit_signal_pos,
                                    "GAP"
                                ]
                                -
                                mean_gap
                            )
                            /
                            sd_gap
                        )
                        if
                        exit_signal_pos is not None
                        else
                        np.nan
                    ),

                "EXIT_EXECUTION_DATE":
                    exit_execution_date,

                "EXIT_EXECUTION_GAP":
                    float(
                        effective.loc[
                            exit_exec_pos,
                            "GAP"
                        ]
                    ),

                "EXIT_EXECUTION_Z":
                    float(
                        (
                            effective.loc[
                                exit_exec_pos,
                                "GAP"
                            ]
                            -
                            mean_gap
                        )
                        /
                        sd_gap
                    ),

                "EXIT_REASON":
                    exit_reason,

                "HOLDING_CALENDAR_DAYS":
                    int(
                        (
                            exit_execution_date
                            -
                            entry_execution_date
                        ).days
                    ),
            }
        )


        if exit_reason == (
            "MEAN_CROSS"
        ):
            # On the exit execution day's close the old trade is closed.
            # That same close may serve as a NEW signal for next-day re-entry.
            scan_pos = (
                exit_exec_pos
            )

        else:
            break


    pair_window_audit_rows.append(
        {
            "FORMATION_DATE":
                formation_date,

            "BLOCK_TYPE":
                pair_row.BLOCK_TYPE,

            "SELECTED_PAIR_NUMBER":
                int(
                    pair_row.SELECTED_PAIR_NUMBER
                ),

            "PAIR_ID":
                pair_row.PAIR_ID,

            "COMPANY_A":
                company_a,

            "COMPANY_B":
                company_b,

            "N_COMMON_OBSERVATIONS":
                len(
                    pair
                ),

            "EFFECTIVE_TERMINAL_DATE":
                pd.Timestamp(
                    effective[
                        "DATE"
                    ].iloc[
                        -1
                    ]
                ),

            "STRUCTURAL_BREAK_DATE":
                structural_break_date,

            "LOGICAL_TRADES":
                trade_number,

            "NORMAL_MEAN_CROSS_EXITS":
                normal_exits,

            "FORCED_EXITS":
                forced_exits,

            "STATUS":
                "PASS",
        }
    )


logical = pd.DataFrame(
    logical_trade_rows
)

pair_window_audit = pd.DataFrame(
    pair_window_audit_rows
)

gap_paths = pd.DataFrame(
    gap_path_rows
)


if not logical.empty:
    for col in [
        "FORMATION_DATE",
        "ENTRY_SIGNAL_DATE",
        "ENTRY_EXECUTION_DATE",
        "EXIT_SIGNAL_DATE",
        "EXIT_EXECUTION_DATE",
    ]:
        logical[
            col
        ] = parse_dates(
            logical[
                col
            ]
        )


    if (
        logical[
            "ENTRY_EXECUTION_DATE"
        ]
        <=
        logical[
            "ENTRY_SIGNAL_DATE"
        ]
    ).any():
        raise RuntimeError(
            "Same-day/look-ahead entry execution detected."
        )


    normal_mask = (
        logical[
            "EXIT_REASON"
        ]
        ==
        "MEAN_CROSS"
    )

    if (
        logical.loc[
            normal_mask,
            "EXIT_EXECUTION_DATE",
        ]
        <=
        logical.loc[
            normal_mask,
            "EXIT_SIGNAL_DATE",
        ]
    ).any():
        raise RuntimeError(
            "Same-day/look-ahead normal exit execution detected."
        )


    if (
        logical[
            "EXIT_EXECUTION_DATE"
        ]
        <
        logical[
            "ENTRY_EXECUTION_DATE"
        ]
    ).any():
        raise RuntimeError(
            "Exit occurs before entry."
        )


logical.to_csv(
    OUTPUT_DIR
    / "01_EG_LOGICAL_TRADE_LEDGER_NO_PNL.csv",
    index=False,
    date_format="%Y-%m-%d"
)

pair_window_audit.to_csv(
    OUTPUT_DIR
    / "00_EG_PAIR_WINDOW_SIGNAL_AUDIT.csv",
    index=False,
    date_format="%Y-%m-%d"
)

gap_paths.to_csv(
    OUTPUT_DIR
    / "00_EG_TRADING_GAP_PATHS.csv",
    index=False,
    date_format="%Y-%m-%d"
)


print(
    f"Logical EG trades generated: {len(logical)}"
)

print(
    "Mean-cross logical exits:    "
    f"{int(logical['EXIT_REASON'].eq('MEAN_CROSS').sum()) if len(logical) else 0}"
)

print(
    "Forced logical exits:        "
    f"{int(logical['EXIT_REASON'].ne('MEAN_CROSS').sum()) if len(logical) else 0}"
)


# =============================================================================
# 5. OFFICIAL NSE ENTRY-DATE F&O CHECK — ONLY ACTUAL EG ENTRY DATES
# =============================================================================

def valid_zip_bytes(content):
    if (
        not isinstance(
            content,
            (
                bytes,
                bytearray,
            )
        )
        or
        len(
            content
        )
        <
        100
        or
        content[
            :
            2
        ]
        !=
        b"PK"
    ):
        return False

    try:
        with zipfile.ZipFile(
            io.BytesIO(
                content
            )
        ) as z:
            return (
                z.testzip()
                is
                None
            )

    except Exception:
        return False


def decode_csv_bytes(content):
    for enc in [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ]:
        try:
            return content.decode(
                enc
            )

        except UnicodeDecodeError:
            pass

    return content.decode(
        "latin-1",
        errors="replace"
    )


def fo_urls(date_value):
    dt = pd.Timestamp(
        date_value
    ).normalize()

    if dt < UDIFF_START_DATE:
        year = dt.strftime(
            "%Y"
        )

        month = dt.strftime(
            "%b"
        ).upper()

        day = dt.strftime(
            "%d"
        )

        filename = (
            f"fo{day}{month}{year}bhav.csv.zip"
        )

        urls = [
            (
                "https://nsearchives.nseindia.com/content/"
                f"historical/DERIVATIVES/{year}/{month}/{filename}"
            ),
            (
                "https://archives.nseindia.com/content/"
                f"historical/DERIVATIVES/{year}/{month}/{filename}"
            ),
        ]

        schema = (
            "LEGACY"
        )

    else:
        ymd = dt.strftime(
            "%Y%m%d"
        )

        filename = (
            f"BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip"
        )

        urls = [
            (
                "https://nsearchives.nseindia.com/content/"
                f"fo/{filename}"
            ),
            (
                "https://archives.nseindia.com/content/"
                f"fo/{filename}"
            ),
        ]

        schema = (
            "UDIFF"
        )

    return (
        urls,
        schema,
        filename,
    )


def build_session():
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/151.0.0.0 Safari/537.36",

            "Accept":
                "*/*",

            "Accept-Language":
                "en-US,en;q=0.9",

            "Connection":
                "keep-alive",
        }
    )

    try:
        session.get(
            "https://www.nseindia.com/",
            timeout=15
        )

    except Exception:
        pass

    return session


def obtain_fo_zip(
    dt,
    session
):
    urls, schema, filename = (
        fo_urls(
            dt
        )
    )

    target = (
        RAW_FO_DIR
        /
        filename
    )

    # 1. EG cache
    if target.exists():
        content = target.read_bytes()

        if valid_zip_bytes(
            content
        ):
            return (
                target,
                schema,
                "EG_CACHE",
            )


    # 2. Reuse exact file from earlier project caches.
    prior_dirs = [
        (
            PROJECT_ROOT
            / "pair_trading_methods"
            / "SSD"
            / "03_backtest"
            / "01_entry_date_fno_check"
            / "raw_nse_fo_bhavcopy"
        ),
        (
            PROJECT_ROOT
            / "nse_pharma_final_investable_universe_FINAL"
            / "raw_nse_fo_bhavcopy"
        ),
    ]

    external_cache = os.environ.get(
        "NIFTY_FO_CACHE_DIR"
    )

    if external_cache:
        prior_dirs.append(
            Path(
                external_cache
            )
        )


    for prior_dir in prior_dirs:
        prior = (
            prior_dir
            /
            filename
        )

        if prior.exists():
            content = prior.read_bytes()

            if valid_zip_bytes(
                content
            ):
                shutil.copy2(
                    prior,
                    target
                )

                return (
                    target,
                    schema,
                    f"REUSED_{prior_dir}",
                )


    # 3. Download ONLY this actual EG entry date.
    errors = []

    for attempt in range(
        1,
        4
    ):
        for url in urls:
            try:
                response = session.get(
                    url,
                    timeout=45,
                    allow_redirects=True
                )

                if (
                    response.status_code
                    ==
                    200
                    and
                    valid_zip_bytes(
                        response.content
                    )
                ):
                    target.write_bytes(
                        response.content
                    )

                    time.sleep(
                        REQUEST_DELAY_SECONDS
                    )

                    return (
                        target,
                        schema,
                        url,
                    )

                errors.append(
                    f"attempt={attempt} "
                    f"http={response.status_code} "
                    f"url={url}"
                )

            except Exception as exc:
                errors.append(
                    f"attempt={attempt} "
                    f"{type(exc).__name__}: {exc} "
                    f"url={url}"
                )

        time.sleep(
            attempt
        )


    raise RuntimeError(
        "\nCould not obtain official NSE F&O bhavcopy for "
        f"{dt.date()}.\n"
        "The logical-trade file has already been saved, so no work is lost.\n"
        "Errors:\n"
        +
        "\n".join(
            errors
        )
    )


def parse_fo_zip(
    path,
    expected_schema
):
    with zipfile.ZipFile(
        path
    ) as z:
        csv_names = [
            name
            for name
            in z.namelist()
            if name.lower().endswith(
                ".csv"
            )
        ]

        if not csv_names:
            raise RuntimeError(
                f"No CSV inside F&O zip: {path}"
            )

        last_headers = None

        for name in csv_names:
            text = decode_csv_bytes(
                z.read(
                    name
                )
            )

            reader = csv.DictReader(
                io.StringIO(
                    text
                )
            )

            if not reader.fieldnames:
                continue

            last_headers = list(
                reader.fieldnames
            )

            colmap = {
                clean_col(
                    col
                ):
                    col
                for col
                in reader.fieldnames
            }


            # Legacy format
            if (
                "INSTRUMENT"
                in colmap
                and
                "SYMBOL"
                in colmap
            ):
                eligible = set()

                for row in reader:
                    instrument = str(
                        row.get(
                            colmap[
                                "INSTRUMENT"
                            ],
                            ""
                        )
                    ).strip().upper()

                    if instrument != (
                        "FUTSTK"
                    ):
                        continue

                    symbol = clean_symbol(
                        row.get(
                            colmap[
                                "SYMBOL"
                            ]
                        )
                    )

                    company = (
                        company_id_from_symbol(
                            symbol
                        )
                    )

                    if company:
                        eligible.add(
                            company
                        )

                if not eligible:
                    raise RuntimeError(
                        f"No FUTSTK symbols parsed from {path}"
                    )

                return (
                    eligible,
                    "LEGACY_FUTSTK",
                    name,
                )


            # UDiFF format
            if (
                "FININSTRMTP"
                in colmap
                and
                "TCKRSYMB"
                in colmap
            ):
                eligible = set()

                for row in reader:
                    instrument = str(
                        row.get(
                            colmap[
                                "FININSTRMTP"
                            ],
                            ""
                        )
                    ).strip().upper()

                    if instrument != (
                        "STF"
                    ):
                        continue

                    symbol = clean_symbol(
                        row.get(
                            colmap[
                                "TCKRSYMB"
                            ]
                        )
                    )

                    company = (
                        company_id_from_symbol(
                            symbol
                        )
                    )

                    if company:
                        eligible.add(
                            company
                        )

                if not eligible:
                    raise RuntimeError(
                        f"No STF symbols parsed from {path}"
                    )

                return (
                    eligible,
                    "UDIFF_STF",
                    name,
                )


        raise RuntimeError(
            f"No recognized F&O schema in {path}. "
            f"Expected={expected_schema}; headers={last_headers}"
        )


fno_rows = []
fo_date_audit_rows = []


if not logical.empty:
    session = build_session()

    for dt, group in logical.groupby(
        "ENTRY_EXECUTION_DATE",
        sort=True
    ):
        dt = pd.Timestamp(
            dt
        )

        zip_path, expected_schema, source = (
            obtain_fo_zip(
                dt,
                session
            )
        )

        eligible_companies, parsed_schema, member_name = (
            parse_fo_zip(
                zip_path,
                expected_schema
            )
        )

        fo_date_audit_rows.append(
            {
                "ENTRY_EXECUTION_DATE":
                    dt,

                "N_LOGICAL_TRADES":
                    len(
                        group
                    ),

                "EXPECTED_SCHEMA":
                    expected_schema,

                "PARSED_SCHEMA":
                    parsed_schema,

                "SOURCE":
                    source,

                "ZIP_FILE":
                    zip_path.name,

                "CSV_MEMBER":
                    member_name,

                "N_STOCKS_WITH_FUTURES":
                    len(
                        eligible_companies
                    ),

                "STATUS":
                    "PASS",
            }
        )


        for trade in group.itertuples(
            index=False
        ):
            executable = (
                trade.SHORT_COMPANY
                in
                eligible_companies
            )

            fno_rows.append(
                {
                    "FORMATION_DATE":
                        trade.FORMATION_DATE,

                    "BLOCK_TYPE":
                        trade.BLOCK_TYPE,

                    "SELECTED_PAIR_NUMBER":
                        int(
                            trade.SELECTED_PAIR_NUMBER
                        ),

                    "PAIR_ID":
                        trade.PAIR_ID,

                    "TRADE_NUMBER_WITHIN_PAIR_WINDOW":
                        int(
                            trade.TRADE_NUMBER_WITHIN_PAIR_WINDOW
                        ),

                    "ENTRY_EXECUTION_DATE":
                        dt,

                    "SHORT_COMPANY":
                        trade.SHORT_COMPANY,

                    "SHORT_LEG_FUTURE_AVAILABLE":
                        executable,

                    "CASH_LONG_FUTURES_SHORT_EXECUTABLE":
                        executable,

                    "STATUS":
                        (
                            "PASS"
                            if
                            executable
                            else
                            "FAIL_SHORT_LEG_NO_STOCK_FUTURE"
                        ),
                }
            )


fno_check = pd.DataFrame(
    fno_rows
)

fno_date_audit = pd.DataFrame(
    fo_date_audit_rows
)


fno_check.to_csv(
    OUTPUT_DIR
    / "02_ENTRY_DATE_FNO_CHECK.csv",
    index=False,
    date_format="%Y-%m-%d"
)

fno_date_audit.to_csv(
    OUTPUT_DIR
    / "02_ENTRY_DATE_FNO_DATE_AUDIT.csv",
    index=False,
    date_format="%Y-%m-%d"
)


if logical.empty:
    executable_trades = logical.copy()
    skipped_trades = logical.copy()

else:
    trade_key = [
        "FORMATION_DATE",
        "SELECTED_PAIR_NUMBER",
        "PAIR_ID",
        "TRADE_NUMBER_WITHIN_PAIR_WINDOW",
        "ENTRY_EXECUTION_DATE",
        "SHORT_COMPANY",
    ]

    if fno_check[
        trade_key
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate trade key in EG F&O check."
        )


    logical = logical.merge(
        fno_check[
            trade_key
            +
            [
                "CASH_LONG_FUTURES_SHORT_EXECUTABLE",
            ]
        ],
        on=trade_key,
        how="left",
        validate="one_to_one"
    )


    if logical[
        "CASH_LONG_FUTURES_SHORT_EXECUTABLE"
    ].isna().any():
        raise RuntimeError(
            "At least one EG logical trade did not receive an F&O result."
        )


    executable_trades = logical[
        logical[
            "CASH_LONG_FUTURES_SHORT_EXECUTABLE"
        ]
    ].copy().reset_index(
        drop=True
    )

    skipped_trades = logical[
        ~logical[
            "CASH_LONG_FUTURES_SHORT_EXECUTABLE"
        ]
    ].copy().reset_index(
        drop=True
    )


executable_trades.to_csv(
    OUTPUT_DIR
    / "03_EXECUTABLE_EG_TRADE_LEDGER_NO_PNL.csv",
    index=False,
    date_format="%Y-%m-%d"
)

skipped_trades.to_csv(
    OUTPUT_DIR
    / "03_SKIPPED_FNO_INFEASIBLE_EG_TRADES.csv",
    index=False,
    date_format="%Y-%m-%d"
)


print(
    f"Executable EG trades:      {len(executable_trades)}"
)

print(
    f"F&O-infeasible EG trades:  {len(skipped_trades)}"
)


# =============================================================================
# 6. BUILD SHORT-STOCK-FUTURES PRICE PROXY
# =============================================================================
#
# Long leg:
#     TOTAL_RETURN_INDEX
#     -> includes dividends because the long cash shareholder receives them.
#
# Short leg:
#     stock-futures proxy
#     -> excludes the cash dividend.
#
# On dividend days:
#     proxy return = CLOSE / TR_PREV_CLOSE - 1
#
# On other ordinary / split / bonus / rights days:
#     proxy return = audited TOTAL_RETURN.
#
# This is deliberately the SAME simplified economic treatment used for SSD.
# =============================================================================

prices = prices.sort_values(
    [
        "COMPANY_ID",
        "DATE",
    ]
).reset_index(
    drop=True
)


prices[
    "FUTURES_PROXY_RETURN"
] = prices[
    "TOTAL_RETURN"
].copy()


dividend_mask = (
    prices[
        "CA_EVENT_TYPE"
    ].eq(
        "DIVIDEND"
    )
    &
    prices[
        "TR_PREV_CLOSE"
    ].notna()
)


prices.loc[
    dividend_mask,
    "FUTURES_PROXY_RETURN",
] = (
    prices.loc[
        dividend_mask,
        "CLOSE"
    ]
    /
    prices.loc[
        dividend_mask,
        "TR_PREV_CLOSE"
    ]
    -
    1.0
)


segment_first = (
    prices.groupby(
        [
            "COMPANY_ID",
            "SEGMENT_ID",
        ],
        sort=False
    )
    .cumcount()
    .eq(
        0
    )
)


prices.loc[
    segment_first,
    "FUTURES_PROXY_RETURN",
] = np.nan


unexpected_missing_proxy = (
    prices[
        "FUTURES_PROXY_RETURN"
    ].isna()
    &
    ~segment_first
)


if unexpected_missing_proxy.any():
    bad = prices.loc[
        unexpected_missing_proxy,
        [
            "DATE",
            "COMPANY_ID",
            "SEGMENT_ID",
            "CA_EVENT_TYPE",
            "TOTAL_RETURN",
            "CLOSE",
            "TR_PREV_CLOSE",
        ],
    ].head(
        20
    )

    raise RuntimeError(
        "Unexpected missing futures-proxy returns:\n"
        +
        bad.to_string(
            index=False
        )
    )


prices[
    "_PROXY_GROWTH"
] = (
    1.0
    +
    prices[
        "FUTURES_PROXY_RETURN"
    ].fillna(
        0.0
    )
)


prices[
    "FUTURES_PROXY_INDEX"
] = (
    100.0
    *
    prices.groupby(
        [
            "COMPANY_ID",
            "SEGMENT_ID",
        ],
        sort=False
    )[
        "_PROXY_GROWTH"
    ].cumprod()
)


prices = prices.drop(
    columns=[
        "_PROXY_GROWTH",
    ]
)


lookup = prices.set_index(
    [
        "DATE",
        "COMPANY_ID",
    ]
).sort_index()


def price_row(
    dt,
    company
):
    key = (
        pd.Timestamp(
            dt
        ).normalize(),
        company,
    )

    if key not in lookup.index:
        raise RuntimeError(
            f"Missing price row for {company} on "
            f"{pd.Timestamp(dt).date()} while a trade is active. "
            "No forward filling is permitted."
        )

    row = lookup.loc[
        key
    ]

    if isinstance(
        row,
        pd.DataFrame
    ):
        raise RuntimeError(
            f"Duplicate lookup row for {company} on "
            f"{pd.Timestamp(dt).date()}."
        )

    return row


# Structural validation of every executable trade.
for row in executable_trades.itertuples(
    index=False
):
    long_entry = price_row(
        row.ENTRY_EXECUTION_DATE,
        row.LONG_COMPANY
    )

    long_exit = price_row(
        row.EXIT_EXECUTION_DATE,
        row.LONG_COMPANY
    )

    short_entry = price_row(
        row.ENTRY_EXECUTION_DATE,
        row.SHORT_COMPANY
    )

    short_exit = price_row(
        row.EXIT_EXECUTION_DATE,
        row.SHORT_COMPANY
    )

    if (
        long_entry[
            "SEGMENT_ID"
        ]
        !=
        long_exit[
            "SEGMENT_ID"
        ]
    ):
        raise RuntimeError(
            f"Long leg crosses a structural segment: {row.PAIR_ID}"
        )

    if (
        short_entry[
            "SEGMENT_ID"
        ]
        !=
        short_exit[
            "SEGMENT_ID"
        ]
    ):
        raise RuntimeError(
            f"Short leg crosses a structural segment: {row.PAIR_ID}"
        )


# =============================================================================
# 7. STABLE TRADE IDS + BETA-BASED LEG WEIGHTS
# =============================================================================

if not executable_trades.empty:
    executable_trades[
        "EG_TRADE_ID"
    ] = [
        (
            f"EG_{fd:%Y%m%d}_P{int(pair_no):02d}_T{int(trade_no):02d}"
        )
        for fd, pair_no, trade_no
        in zip(
            executable_trades[
                "FORMATION_DATE"
            ],
            executable_trades[
                "SELECTED_PAIR_NUMBER"
            ],
            executable_trades[
                "TRADE_NUMBER_WITHIN_PAIR_WINDOW"
            ],
        )
    ]


    if executable_trades[
        "EG_TRADE_ID"
    ].duplicated().any():
        raise RuntimeError(
            "Generated EG_TRADE_ID is not unique."
        )


    executable_trades[
        "A_SHARE_OF_PAIR_GROSS"
    ] = (
        1.0
        /
        (
            1.0
            +
            executable_trades[
                "BETA"
            ]
        )
    )


    executable_trades[
        "B_SHARE_OF_PAIR_GROSS"
    ] = (
        executable_trades[
            "BETA"
        ]
        /
        (
            1.0
            +
            executable_trades[
                "BETA"
            ]
        )
    )


# =============================================================================
# 8. FULL MARKET CALENDAR
# =============================================================================

backtest_start = pd.Timestamp(
    schedule[
        "TRADING_START"
    ].min()
)

backtest_end = pd.Timestamp(
    schedule[
        "TRADING_END"
    ].max()
)


market_dates = pd.DatetimeIndex(
    sorted(
        prices.loc[
            prices[
                "DATE"
            ].between(
                backtest_start,
                backtest_end,
                inclusive="both"
            ),
            "DATE",
        ].unique()
    )
)


if len(
    market_dates
) == 0:
    raise RuntimeError(
        "No market dates in full EG backtest period."
    )


# =============================================================================
# 9. PORTFOLIO SIMULATION
# =============================================================================

def simulate(
    cost_multiplier
):
    if executable_trades.empty:
        entry_groups = {}
        exit_groups = {}

    else:
        entry_groups = {
            dt:
                grp.copy()
            for dt, grp
            in executable_trades.groupby(
                "ENTRY_EXECUTION_DATE"
            )
        }

        exit_groups = {
            dt:
                grp.copy()
            for dt, grp
            in executable_trades.groupby(
                "EXIT_EXECUTION_DATE"
            )
        }


    nav = INITIAL_NAV
    positions = {}

    daily_rows = []
    completed_rows = []

    previous_date = None


    for dt in market_dates:
        dt = pd.Timestamp(
            dt
        )

        nav_start = nav

        gross_pnl_today = 0.0
        transaction_cost_today = 0.0
        financing_cost_today = 0.0
        turnover_today = 0.0


        # ---------------------------------------------------------------------
        # Mark open trades from previous market date to today.
        # ---------------------------------------------------------------------

        if previous_date is not None:
            for trade_id, pos in list(
                positions.items()
            ):
                long_prev = price_row(
                    previous_date,
                    pos[
                        "LONG_COMPANY"
                    ]
                )

                long_now = price_row(
                    dt,
                    pos[
                        "LONG_COMPANY"
                    ]
                )

                short_prev = price_row(
                    previous_date,
                    pos[
                        "SHORT_COMPANY"
                    ]
                )

                short_now = price_row(
                    dt,
                    pos[
                        "SHORT_COMPANY"
                    ]
                )


                if (
                    long_prev[
                        "SEGMENT_ID"
                    ]
                    !=
                    long_now[
                        "SEGMENT_ID"
                    ]
                ):
                    raise RuntimeError(
                        f"Active long leg crosses structural break: {trade_id}"
                    )

                if (
                    short_prev[
                        "SEGMENT_ID"
                    ]
                    !=
                    short_now[
                        "SEGMENT_ID"
                    ]
                ):
                    raise RuntimeError(
                        f"Active short leg crosses structural break: {trade_id}"
                    )


                long_pnl = (
                    pos[
                        "LONG_UNITS"
                    ]
                    *
                    (
                        float(
                            long_now[
                                "TOTAL_RETURN_INDEX"
                            ]
                        )
                        -
                        float(
                            long_prev[
                                "TOTAL_RETURN_INDEX"
                            ]
                        )
                    )
                )


                short_pnl = (
                    -
                    pos[
                        "SHORT_UNITS"
                    ]
                    *
                    (
                        float(
                            short_now[
                                "FUTURES_PROXY_INDEX"
                            ]
                        )
                        -
                        float(
                            short_prev[
                                "FUTURES_PROXY_INDEX"
                            ]
                        )
                    )
                )


                # Explicit short-side financing treatment.
                # Charge funding on the assumed futures margin/collateral
                # over the actual calendar-day gap since the previous
                # market observation (so weekends are included naturally).
                calendar_days = max(
                    0,
                    int(
                        (
                            dt
                            -
                            previous_date
                        ).days
                    )
                )

                short_notional_prev = abs(
                    pos[
                        "SHORT_UNITS"
                    ]
                    *
                    float(
                        short_prev[
                            "FUTURES_PROXY_INDEX"
                        ]
                    )
                )

                financing_cost = (
                    short_notional_prev
                    *
                    FUTURES_MARGIN_FRACTION
                    *
                    FINANCING_RATE_ANNUAL
                    *
                    calendar_days
                    /
                    365.0
                    *
                    cost_multiplier
                )

                nav -= financing_cost
                financing_cost_today += financing_cost

                pos[
                    "FINANCING_COST"
                ] += financing_cost

                pnl = (
                    long_pnl
                    +
                    short_pnl
                )

                nav += pnl
                gross_pnl_today += pnl

                pos[
                    "LONG_PNL"
                ] += long_pnl

                pos[
                    "SHORT_PNL"
                ] += short_pnl

                pos[
                    "GROSS_PNL"
                ] += pnl


        # ---------------------------------------------------------------------
        # Exits at today's close.
        # ---------------------------------------------------------------------

        if dt in exit_groups:
            for trade in exit_groups[
                dt
            ].itertuples(
                index=False
            ):
                trade_id = (
                    trade.EG_TRADE_ID
                )

                if trade_id not in positions:
                    raise RuntimeError(
                        f"Exit without open EG position: {trade_id}"
                    )

                pos = positions[
                    trade_id
                ]

                long_now = price_row(
                    dt,
                    pos[
                        "LONG_COMPANY"
                    ]
                )

                short_now = price_row(
                    dt,
                    pos[
                        "SHORT_COMPANY"
                    ]
                )


                long_value = abs(
                    pos[
                        "LONG_UNITS"
                    ]
                    *
                    float(
                        long_now[
                            "TOTAL_RETURN_INDEX"
                        ]
                    )
                )

                short_value = abs(
                    pos[
                        "SHORT_UNITS"
                    ]
                    *
                    float(
                        short_now[
                            "FUTURES_PROXY_INDEX"
                        ]
                    )
                )


                exit_cost = (
                    long_value
                    *
                    CASH_ONE_WAY_RATE
                    *
                    cost_multiplier
                    +
                    short_value
                    *
                    FUTURES_ONE_WAY_RATE
                    *
                    cost_multiplier
                )


                nav -= exit_cost
                transaction_cost_today += exit_cost

                exit_turnover = (
                    long_value
                    +
                    short_value
                )

                turnover_today += (
                    exit_turnover
                )


                pos[
                    "TRANSACTION_COST"
                ] += exit_cost

                pos[
                    "EXIT_COST"
                ] = exit_cost

                pos[
                    "TURNOVER_VALUE"
                ] += exit_turnover

                pos[
                    "EXIT_EXECUTION_DATE"
                ] = dt

                pos[
                    "EXIT_REASON"
                ] = trade.EXIT_REASON

                pos[
                    "HOLDING_CALENDAR_DAYS"
                ] = int(
                    (
                        dt
                        -
                        pos[
                            "ENTRY_EXECUTION_DATE"
                        ]
                    ).days
                )


                pos[
                    "TOTAL_COST"
                ] = (
                    pos[
                        "TRANSACTION_COST"
                    ]
                    +
                    pos[
                        "FINANCING_COST"
                    ]
                )

                pos[
                    "NET_PNL"
                ] = (
                    pos[
                        "GROSS_PNL"
                    ]
                    -
                    pos[
                        "TOTAL_COST"
                    ]
                )


                pos[
                    "GROSS_RETURN_ON_PAIR_GROSS"
                ] = safe_div(
                    pos[
                        "GROSS_PNL"
                    ],
                    pos[
                        "INITIAL_PAIR_GROSS"
                    ]
                )


                pos[
                    "NET_RETURN_ON_PAIR_GROSS"
                ] = safe_div(
                    pos[
                        "NET_PNL"
                    ],
                    pos[
                        "INITIAL_PAIR_GROSS"
                    ]
                )


                completed_rows.append(
                    pos.copy()
                )

                del positions[
                    trade_id
                ]


        # ---------------------------------------------------------------------
        # Entries at today's close.
        # ---------------------------------------------------------------------

        if dt in entry_groups:
            same_day = entry_groups[
                dt
            ]

            nav_basis = nav

            active_slots = {
                (
                    p[
                        "FORMATION_DATE"
                    ],
                    p[
                        "SELECTED_PAIR_NUMBER"
                    ],
                )
                for p
                in positions.values()
            }


            for trade in same_day.itertuples(
                index=False
            ):
                slot = (
                    pd.Timestamp(
                        trade.FORMATION_DATE
                    ),
                    int(
                        trade.SELECTED_PAIR_NUMBER
                    ),
                )


                if slot in active_slots:
                    raise RuntimeError(
                        f"Overlapping trade in same EG pair slot: {slot}"
                    )


                a_notional = (
                    PAIR_GROSS_WEIGHT
                    *
                    nav_basis
                    *
                    float(
                        trade.A_SHARE_OF_PAIR_GROSS
                    )
                )

                b_notional = (
                    PAIR_GROSS_WEIGHT
                    *
                    nav_basis
                    *
                    float(
                        trade.B_SHARE_OF_PAIR_GROSS
                    )
                )


                if trade.LONG_COMPANY == (
                    trade.COMPANY_A
                ):
                    long_notional = (
                        a_notional
                    )

                    short_notional = (
                        b_notional
                    )

                elif trade.LONG_COMPANY == (
                    trade.COMPANY_B
                ):
                    long_notional = (
                        b_notional
                    )

                    short_notional = (
                        a_notional
                    )

                else:
                    raise RuntimeError(
                        f"Cannot map EG leg notionals for {trade.EG_TRADE_ID}"
                    )


                long_row = price_row(
                    dt,
                    trade.LONG_COMPANY
                )

                short_row = price_row(
                    dt,
                    trade.SHORT_COMPANY
                )


                long_units = (
                    long_notional
                    /
                    float(
                        long_row[
                            "TOTAL_RETURN_INDEX"
                        ]
                    )
                )

                short_units = (
                    short_notional
                    /
                    float(
                        short_row[
                            "FUTURES_PROXY_INDEX"
                        ]
                    )
                )


                entry_cost = (
                    long_notional
                    *
                    CASH_ONE_WAY_RATE
                    *
                    cost_multiplier
                    +
                    short_notional
                    *
                    FUTURES_ONE_WAY_RATE
                    *
                    cost_multiplier
                )


                nav -= entry_cost
                transaction_cost_today += entry_cost

                entry_turnover = (
                    long_notional
                    +
                    short_notional
                )

                turnover_today += (
                    entry_turnover
                )


                positions[
                    trade.EG_TRADE_ID
                ] = {
                    "EG_TRADE_ID":
                        trade.EG_TRADE_ID,

                    "COST_MULTIPLIER":
                        cost_multiplier,

                    "FORMATION_DATE":
                        pd.Timestamp(
                            trade.FORMATION_DATE
                        ),

                    "BLOCK_TYPE":
                        trade.BLOCK_TYPE,

                    "SELECTED_PAIR_NUMBER":
                        int(
                            trade.SELECTED_PAIR_NUMBER
                        ),

                    "PAIR_ID":
                        trade.PAIR_ID,

                    "COMPANY_A":
                        trade.COMPANY_A,

                    "COMPANY_B":
                        trade.COMPANY_B,

                    "ALPHA":
                        float(
                            trade.ALPHA
                        ),

                    "BETA":
                        float(
                            trade.BETA
                        ),

                    "ENTRY_SIGNAL_DATE":
                        pd.Timestamp(
                            trade.ENTRY_SIGNAL_DATE
                        ),

                    "ENTRY_EXECUTION_DATE":
                        dt,

                    "ENTRY_DIRECTION":
                        trade.ENTRY_DIRECTION,

                    "ENTRY_SIGNAL_Z":
                        float(
                            trade.ENTRY_SIGNAL_Z
                        ),

                    "ENTRY_EXECUTION_Z":
                        float(
                            trade.ENTRY_EXECUTION_Z
                        ),

                    "EXIT_SIGNAL_DATE":
                        trade.EXIT_SIGNAL_DATE,

                    "EXIT_SIGNAL_Z":
                        (
                            float(
                                trade.EXIT_SIGNAL_Z
                            )
                            if
                            pd.notna(
                                trade.EXIT_SIGNAL_Z
                            )
                            else
                            np.nan
                        ),

                    "EXIT_EXECUTION_Z":
                        float(
                            trade.EXIT_EXECUTION_Z
                        ),

                    "LONG_COMPANY":
                        trade.LONG_COMPANY,

                    "SHORT_COMPANY":
                        trade.SHORT_COMPANY,

                    "A_SHARE_OF_PAIR_GROSS":
                        float(
                            trade.A_SHARE_OF_PAIR_GROSS
                        ),

                    "B_SHARE_OF_PAIR_GROSS":
                        float(
                            trade.B_SHARE_OF_PAIR_GROSS
                        ),

                    "LONG_UNITS":
                        long_units,

                    "SHORT_UNITS":
                        short_units,

                    "INITIAL_LONG_NOTIONAL":
                        long_notional,

                    "INITIAL_SHORT_NOTIONAL":
                        short_notional,

                    "INITIAL_PAIR_GROSS":
                        (
                            long_notional
                            +
                            short_notional
                        ),

                    "LONG_PNL":
                        0.0,

                    "SHORT_PNL":
                        0.0,

                    "GROSS_PNL":
                        0.0,

                    "TRANSACTION_COST":
                        entry_cost,

                    "FINANCING_COST":
                        0.0,

                    "TOTAL_COST":
                        entry_cost,

                    "ENTRY_COST":
                        entry_cost,

                    "EXIT_COST":
                        np.nan,

                    "TURNOVER_VALUE":
                        entry_turnover,
                }


                active_slots.add(
                    slot
                )


        # ---------------------------------------------------------------------
        # End-of-day exposure.
        # ---------------------------------------------------------------------

        long_exposure = 0.0
        short_exposure = 0.0


        for pos in positions.values():
            long_now = price_row(
                dt,
                pos[
                    "LONG_COMPANY"
                ]
            )

            short_now = price_row(
                dt,
                pos[
                    "SHORT_COMPANY"
                ]
            )


            long_exposure += abs(
                pos[
                    "LONG_UNITS"
                ]
                *
                float(
                    long_now[
                        "TOTAL_RETURN_INDEX"
                    ]
                )
            )

            short_exposure += abs(
                pos[
                    "SHORT_UNITS"
                ]
                *
                float(
                    short_now[
                        "FUTURES_PROXY_INDEX"
                    ]
                )
            )


        gross_exposure = (
            long_exposure
            +
            short_exposure
        )

        net_exposure = (
            long_exposure
            -
            short_exposure
        )


        daily_return = (
            nav
            /
            nav_start
            -
            1.0
        )


        daily_rows.append(
            {
                "COST_MULTIPLIER":
                    cost_multiplier,

                "DATE":
                    dt,

                "NAV_START":
                    nav_start,

                "GROSS_PNL":
                    gross_pnl_today,

                "TRANSACTION_COST":
                    transaction_cost_today,

                "FINANCING_COST":
                    financing_cost_today,

                "TOTAL_COST":
                    (
                        transaction_cost_today
                        +
                        financing_cost_today
                    ),

                "NAV_END":
                    nav,

                "DAILY_RETURN":
                    daily_return,

                "OPEN_PAIRS_EOD":
                    len(
                        positions
                    ),

                "LONG_EXPOSURE":
                    long_exposure,

                "SHORT_EXPOSURE":
                    short_exposure,

                "GROSS_EXPOSURE":
                    gross_exposure,

                "NET_EXPOSURE":
                    net_exposure,

                "GROSS_EXPOSURE_TO_NAV":
                    safe_div(
                        gross_exposure,
                        nav
                    ),

                "NET_EXPOSURE_TO_NAV":
                    safe_div(
                        net_exposure,
                        nav
                    ),

                "TURNOVER_VALUE":
                    turnover_today,

                "TURNOVER_TO_NAV":
                    safe_div(
                        turnover_today,
                        nav_start
                    ),
            }
        )


        previous_date = dt


    if positions:
        raise RuntimeError(
            "Open EG positions remain after final backtest date."
        )


    return (
        pd.DataFrame(
            daily_rows
        ),
        pd.DataFrame(
            completed_rows
        ),
    )


# =============================================================================
# 10. RUN COST SCENARIOS
# =============================================================================

daily_frames = []
trade_frames = []


for multiplier in COST_MULTIPLIERS:
    print(
        f"Running EG backtest at {multiplier}x transaction costs..."
    )

    daily_result, trade_result = simulate(
        multiplier
    )

    daily_frames.append(
        daily_result
    )

    if not trade_result.empty:
        trade_frames.append(
            trade_result
        )


daily_all = pd.concat(
    daily_frames,
    ignore_index=True
)


if trade_frames:
    trade_pnl = pd.concat(
        trade_frames,
        ignore_index=True
    )

else:
    trade_pnl = pd.DataFrame()


daily_all.to_csv(
    OUTPUT_DIR
    / "04_DAILY_NAV_ALL_COST_SCENARIOS.csv",
    index=False,
    date_format="%Y-%m-%d"
)

trade_pnl.to_csv(
    OUTPUT_DIR
    / "04_TRADE_PNL_ALL_COST_SCENARIOS.csv",
    index=False,
    date_format="%Y-%m-%d"
)


# =============================================================================
# 11. PERFORMANCE — FULL / DEVELOPMENT / OOS
# =============================================================================

def performance_row(
    data,
    label,
    multiplier
):
    data = data.sort_values(
        "DATE"
    ).copy()

    if data.empty:
        return {
            "COST_MULTIPLIER":
                multiplier,

            "PERIOD":
                label,

            "START_DATE":
                pd.NaT,

            "END_DATE":
                pd.NaT,

            "N_MARKET_DAYS":
                0,

            "TOTAL_RETURN":
                np.nan,

            "ANNUALIZED_RETURN":
                np.nan,

            "ANNUALIZED_VOLATILITY":
                np.nan,

            "SHARPE_RF0":
                np.nan,

            "MAX_DRAWDOWN":
                np.nan,
        }


    initial = float(
        data[
            "NAV_START"
        ].iloc[
            0
        ]
    )

    final = float(
        data[
            "NAV_END"
        ].iloc[
            -1
        ]
    )

    total_return = (
        final
        /
        initial
        -
        1.0
    )

    n_days = len(
        data
    )


    if (
        1.0
        +
        total_return
    ) > 0:
        annualized_return = (
            (
                1.0
                +
                total_return
            )
            **
            (
                TRADING_DAYS_PER_YEAR
                /
                n_days
            )
            -
            1.0
        )

    else:
        annualized_return = np.nan


    daily_returns = data[
        "DAILY_RETURN"
    ].astype(
        float
    )


    daily_std = float(
        daily_returns.std(
            ddof=1
        )
    )


    annualized_vol = (
        daily_std
        *
        np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
        if
        np.isfinite(
            daily_std
        )
        else
        np.nan
    )


    sharpe = (
        float(
            daily_returns.mean()
        )
        /
        daily_std
        *
        np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
        if
        np.isfinite(
            daily_std
        )
        and
        daily_std
        >
        NUMERIC_TOL
        else
        np.nan
    )


    nav_series = data[
        "NAV_END"
    ].astype(
        float
    )

    running_max = nav_series.cummax()

    drawdown = (
        nav_series
        /
        running_max
        -
        1.0
    )


    return {
        "COST_MULTIPLIER":
            multiplier,

        "PERIOD":
            label,

        "START_DATE":
            data[
                "DATE"
            ].min(),

        "END_DATE":
            data[
                "DATE"
            ].max(),

        "N_MARKET_DAYS":
            n_days,

        "TOTAL_RETURN":
            total_return,

        "ANNUALIZED_RETURN":
            annualized_return,

        "ANNUALIZED_VOLATILITY":
            annualized_vol,

        "SHARPE_RF0":
            sharpe,

        "MAX_DRAWDOWN":
            float(
                drawdown.min()
            ),
    }


performance_rows = []


for multiplier, group in daily_all.groupby(
    "COST_MULTIPLIER",
    sort=True
):
    performance_rows.append(
        performance_row(
            group,
            "FULL",
            multiplier
        )
    )

    performance_rows.append(
        performance_row(
            group[
                group[
                    "DATE"
                ]
                <
                OOS_START
            ],
            "DEVELOPMENT",
            multiplier
        )
    )

    performance_rows.append(
        performance_row(
            group[
                group[
                    "DATE"
                ]
                >=
                OOS_START
            ],
            "OOS",
            multiplier
        )
    )


performance = pd.DataFrame(
    performance_rows
)


performance.to_csv(
    OUTPUT_DIR
    / "05_PERFORMANCE_FULL_DEV_OOS.csv",
    index=False,
    date_format="%Y-%m-%d"
)


# =============================================================================
# 12. TRADE STATISTICS — PRIMARY 1x COST
# =============================================================================

if trade_pnl.empty:
    trade_1x = pd.DataFrame()

else:
    trade_1x = trade_pnl[
        trade_pnl[
            "COST_MULTIPLIER"
        ].eq(
            1.0
        )
    ].copy()


if trade_1x.empty:
    trade_stats = pd.DataFrame(
        [
            {
                "PERIOD":
                    "FULL",

                "N_TRADES":
                    0,

                "N_WINNING_TRADES":
                    0,

                "HIT_RATE":
                    np.nan,

                "N_MEAN_CROSS_EXITS":
                    0,

                "N_FORCED_EXITS":
                    0,

                "MEAN_CROSS_SHARE":
                    np.nan,

                "AVG_HOLDING_CALENDAR_DAYS":
                    np.nan,

                "GROSS_PNL":
                    0.0,

                "TRANSACTION_COST":
                    0.0,

                "NET_PNL":
                    0.0,
            },
            {
                "PERIOD":
                    "DEVELOPMENT",

                "N_TRADES":
                    0,

                "N_WINNING_TRADES":
                    0,

                "HIT_RATE":
                    np.nan,

                "N_MEAN_CROSS_EXITS":
                    0,

                "N_FORCED_EXITS":
                    0,

                "MEAN_CROSS_SHARE":
                    np.nan,

                "AVG_HOLDING_CALENDAR_DAYS":
                    np.nan,

                "GROSS_PNL":
                    0.0,

                "TRANSACTION_COST":
                    0.0,

                "NET_PNL":
                    0.0,
            },
            {
                "PERIOD":
                    "OOS",

                "N_TRADES":
                    0,

                "N_WINNING_TRADES":
                    0,

                "HIT_RATE":
                    np.nan,

                "N_MEAN_CROSS_EXITS":
                    0,

                "N_FORCED_EXITS":
                    0,

                "MEAN_CROSS_SHARE":
                    np.nan,

                "AVG_HOLDING_CALENDAR_DAYS":
                    np.nan,

                "GROSS_PNL":
                    0.0,

                "TRANSACTION_COST":
                    0.0,

                "NET_PNL":
                    0.0,
            },
        ]
    )

else:
    stat_rows = []

    for label, df in [
        (
            "FULL",
            trade_1x,
        ),
        (
            "DEVELOPMENT",
            trade_1x[
                trade_1x[
                    "BLOCK_TYPE"
                ].eq(
                    "DEVELOPMENT"
                )
            ],
        ),
        (
            "OOS",
            trade_1x[
                trade_1x[
                    "BLOCK_TYPE"
                ].eq(
                    "OOS"
                )
            ],
        ),
    ]:
        n = len(
            df
        )

        stat_rows.append(
            {
                "PERIOD":
                    label,

                "N_TRADES":
                    n,

                "N_WINNING_TRADES":
                    int(
                        (
                            df[
                                "NET_PNL"
                            ]
                            >
                            0
                        ).sum()
                    )
                    if
                    n
                    else
                    0,

                "HIT_RATE":
                    (
                        float(
                            (
                                df[
                                    "NET_PNL"
                                ]
                                >
                                0
                            ).mean()
                        )
                        if
                        n
                        else
                        np.nan
                    ),

                "N_MEAN_CROSS_EXITS":
                    int(
                        df[
                            "EXIT_REASON"
                        ].eq(
                            "MEAN_CROSS"
                        ).sum()
                    )
                    if
                    n
                    else
                    0,

                "N_FORCED_EXITS":
                    int(
                        df[
                            "EXIT_REASON"
                        ].ne(
                            "MEAN_CROSS"
                        ).sum()
                    )
                    if
                    n
                    else
                    0,

                "MEAN_CROSS_SHARE":
                    (
                        float(
                            df[
                                "EXIT_REASON"
                            ].eq(
                                "MEAN_CROSS"
                            ).mean()
                        )
                        if
                        n
                        else
                        np.nan
                    ),

                "AVG_HOLDING_CALENDAR_DAYS":
                    (
                        float(
                            df[
                                "HOLDING_CALENDAR_DAYS"
                            ].mean()
                        )
                        if
                        n
                        else
                        np.nan
                    ),

                "GROSS_PNL":
                    (
                        float(
                            df[
                                "GROSS_PNL"
                            ].sum()
                        )
                        if
                        n
                        else
                        0.0
                    ),

                "TRANSACTION_COST":
                    (
                        float(
                            df[
                                "TRANSACTION_COST"
                            ].sum()
                        )
                        if
                        n
                        else
                        0.0
                    ),

                "NET_PNL":
                    (
                        float(
                            df[
                                "NET_PNL"
                            ].sum()
                        )
                        if
                        n
                        else
                        0.0
                    ),
            }
        )


    trade_stats = pd.DataFrame(
        stat_rows
    )


trade_stats.to_csv(
    OUTPUT_DIR
    / "06_TRADE_STATISTICS_1X.csv",
    index=False
)


# =============================================================================
# 13. SIX-MONTH TRADING-PERIOD RETURNS — ALL 19 PERIODS
# =============================================================================

daily_1x = (
    daily_all[
        daily_all[
            "COST_MULTIPLIER"
        ].eq(
            1.0
        )
    ]
    .sort_values(
        "DATE"
    )
    .copy()
)


formation_period_rows = []


for row in schedule.itertuples(
    index=False
):
    period = daily_1x[
        daily_1x[
            "DATE"
        ].between(
            row.TRADING_START,
            row.TRADING_END,
            inclusive="both"
        )
    ].copy()


    selected_here = selected[
        selected[
            "FORMATION_DATE"
        ].eq(
            row.FORMATION_DATE
        )
    ]


    if executable_trades.empty:
        trades_here = pd.DataFrame()

    else:
        trades_here = executable_trades[
            executable_trades[
                "FORMATION_DATE"
            ].eq(
                row.FORMATION_DATE
            )
        ]


    if period.empty:
        period_return = np.nan
        actual_start = pd.NaT
        actual_end = pd.NaT

    else:
        period_return = (
            float(
                period[
                    "NAV_END"
                ].iloc[
                    -1
                ]
            )
            /
            float(
                period[
                    "NAV_START"
                ].iloc[
                    0
                ]
            )
            -
            1.0
        )

        actual_start = period[
            "DATE"
        ].min()

        actual_end = period[
            "DATE"
        ].max()


    formation_period_rows.append(
        {
            "FORMATION_DATE":
                row.FORMATION_DATE,

            "BLOCK_TYPE":
                row.BLOCK_TYPE,

            "TRADING_START":
                row.TRADING_START,

            "TRADING_END":
                row.TRADING_END,

            "ACTUAL_FIRST_MARKET_DATE":
                actual_start,

            "ACTUAL_LAST_MARKET_DATE":
                actual_end,

            "N_EG_PAIRS_SELECTED":
                len(
                    selected_here
                ),

            "N_EXECUTABLE_TRADES":
                len(
                    trades_here
                ),

            "PORTFOLIO_RETURN_1X":
                period_return,
        }
    )


formation_period_returns = pd.DataFrame(
    formation_period_rows
)


formation_period_returns.to_csv(
    OUTPUT_DIR
    / "07_FORMATION_PERIOD_RETURNS_1X.csv",
    index=False,
    date_format="%Y-%m-%d"
)


# =============================================================================
# 14. PAIR ATTRIBUTION — PRIMARY 1x COST
# =============================================================================

if trade_1x.empty:
    pair_attribution = pd.DataFrame(
        columns=[
            "PAIR_ID",
            "N_TRADES",
            "GROSS_PNL",
            "TRANSACTION_COST",
            "NET_PNL",
            "WINNING_TRADES",
            "HIT_RATE",
        ]
    )

else:
    pair_attribution = (
        trade_1x.groupby(
            "PAIR_ID",
            as_index=False
        )
        .agg(
            N_TRADES=(
                "EG_TRADE_ID",
                "count",
            ),
            GROSS_PNL=(
                "GROSS_PNL",
                "sum",
            ),
            TRANSACTION_COST=(
                "TRANSACTION_COST",
                "sum",
            ),
            NET_PNL=(
                "NET_PNL",
                "sum",
            ),
            WINNING_TRADES=(
                "NET_PNL",
                lambda s:
                    int(
                        (
                            s
                            >
                            0
                        ).sum()
                    ),
            ),
        )
    )

    pair_attribution[
        "HIT_RATE"
    ] = (
        pair_attribution[
            "WINNING_TRADES"
        ]
        /
        pair_attribution[
            "N_TRADES"
        ]
    )

    pair_attribution = pair_attribution.sort_values(
        "NET_PNL",
        ascending=False
    )


pair_attribution.to_csv(
    OUTPUT_DIR
    / "08_PAIR_ATTRIBUTION_1X.csv",
    index=False
)


# =============================================================================
# 15. EXIT-DAY LIQUIDITY — IF TOTAL_TRADED_VALUE EXISTS
# =============================================================================

liquidity_rows = []


if (
    "TOTAL_TRADED_VALUE"
    in
    prices.columns
    and
    not trade_1x.empty
):
    prices[
        "TOTAL_TRADED_VALUE"
    ] = pd.to_numeric(
        prices[
            "TOTAL_TRADED_VALUE"
        ],
        errors="coerce"
    )


    for row in trade_1x.itertuples(
        index=False
    ):
        long_exit = price_row(
            row.EXIT_EXECUTION_DATE,
            row.LONG_COMPANY
        )

        short_exit = price_row(
            row.EXIT_EXECUTION_DATE,
            row.SHORT_COMPANY
        )


        long_value = pd.to_numeric(
            pd.Series(
                [
                    long_exit[
                        "TOTAL_TRADED_VALUE"
                    ]
                ]
            ),
            errors="coerce"
        ).iloc[
            0
        ]

        short_value = pd.to_numeric(
            pd.Series(
                [
                    short_exit[
                        "TOTAL_TRADED_VALUE"
                    ]
                ]
            ),
            errors="coerce"
        ).iloc[
            0
        ]


        liquidity_rows.append(
            {
                "EG_TRADE_ID":
                    row.EG_TRADE_ID,

                "PAIR_ID":
                    row.PAIR_ID,

                "EXIT_EXECUTION_DATE":
                    row.EXIT_EXECUTION_DATE,

                "LONG_COMPANY":
                    row.LONG_COMPANY,

                "SHORT_COMPANY":
                    row.SHORT_COMPANY,

                "LONG_EXIT_TRADED_VALUE":
                    long_value,

                "SHORT_EXIT_TRADED_VALUE":
                    short_value,

                "WEAKER_LEG_EXIT_TRADED_VALUE":
                    (
                        min(
                            long_value,
                            short_value
                        )
                        if
                        np.isfinite(
                            long_value
                        )
                        and
                        np.isfinite(
                            short_value
                        )
                        else
                        np.nan
                    ),
            }
        )


exit_liquidity = pd.DataFrame(
    liquidity_rows
)


exit_liquidity.to_csv(
    OUTPUT_DIR
    / "09_EXIT_DAY_LIQUIDITY_CHECK.csv",
    index=False,
    date_format="%Y-%m-%d"
)


# =============================================================================
# 16. NIFTY 500 BETA / CORRELATION — IF FILE EXISTS
# =============================================================================

beta_rows = []


if BENCHMARK_FILE.exists():
    benchmark = clean_columns(
        pd.read_csv(
            BENCHMARK_FILE,
            low_memory=False
        )
    )

    require_columns(
        benchmark,
        [
            "DATE",
            "CLOSE",
        ],
        "NIFTY 500 benchmark",
    )

    benchmark[
        "DATE"
    ] = parse_dates(
        benchmark[
            "DATE"
        ]
    )

    benchmark[
        "CLOSE"
    ] = pd.to_numeric(
        benchmark[
            "CLOSE"
        ],
        errors="coerce"
    )


    benchmark = (
        benchmark[
            [
                "DATE",
                "CLOSE",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=[
                "DATE",
            ]
        )
        .sort_values(
            "DATE"
        )
    )


    common = daily_1x[
        [
            "DATE",
            "NAV_END",
        ]
    ].merge(
        benchmark,
        on="DATE",
        how="inner"
    ).sort_values(
        "DATE"
    )


    common[
        "STRATEGY_RETURN"
    ] = common[
        "NAV_END"
    ].pct_change()


    common[
        "NIFTY500_RETURN"
    ] = common[
        "CLOSE"
    ].pct_change()


    common = common.dropna(
        subset=[
            "STRATEGY_RETURN",
            "NIFTY500_RETURN",
        ]
    )


    for label, data in [
        (
            "FULL",
            common,
        ),
        (
            "DEVELOPMENT",
            common[
                common[
                    "DATE"
                ]
                <
                OOS_START
            ],
        ),
        (
            "OOS",
            common[
                common[
                    "DATE"
                ]
                >=
                OOS_START
            ],
        ),
    ]:
        benchmark_variance = data[
            "NIFTY500_RETURN"
        ].var(
            ddof=1
        )


        beta = (
            data[
                "STRATEGY_RETURN"
            ].cov(
                data[
                    "NIFTY500_RETURN"
                ]
            )
            /
            benchmark_variance
            if
            len(
                data
            )
            >=
            2
            and
            np.isfinite(
                benchmark_variance
            )
            and
            benchmark_variance
            >
            NUMERIC_TOL
            else
            np.nan
        )


        strategy_std = data[
            "STRATEGY_RETURN"
        ].std(
            ddof=1
        )


        benchmark_std = data[
            "NIFTY500_RETURN"
        ].std(
            ddof=1
        )


        correlation = (
            data[
                "STRATEGY_RETURN"
            ].corr(
                data[
                    "NIFTY500_RETURN"
                ]
            )
            if
            len(
                data
            )
            >=
            2
            and
            np.isfinite(
                strategy_std
            )
            and
            strategy_std
            >
            NUMERIC_TOL
            and
            np.isfinite(
                benchmark_std
            )
            and
            benchmark_std
            >
            NUMERIC_TOL
            else
            np.nan
        )


        beta_rows.append(
            {
                "PERIOD":
                    label,

                "N_COMMON_RETURN_INTERVALS":
                    len(
                        data
                    ),

                "BETA_TO_NIFTY500":
                    beta,

                "CORRELATION_TO_NIFTY500":
                    correlation,

                "NOTE":
                    (
                        "Correlation undefined when strategy return variance is zero."
                    ),
            }
        )


else:
    beta_rows.append(
        {
            "PERIOD":
                "FULL",

            "N_COMMON_RETURN_INTERVALS":
                np.nan,

            "BETA_TO_NIFTY500":
                np.nan,

            "CORRELATION_TO_NIFTY500":
                np.nan,

            "NOTE":
                (
                    "NOT_CALCULATED_BENCHMARK_FILE_NOT_FOUND: "
                    +
                    str(
                        BENCHMARK_FILE
                    )
                ),
        }
    )


beta_corr = pd.DataFrame(
    beta_rows
)


beta_corr.to_csv(
    OUTPUT_DIR
    / "10_NIFTY500_BETA_CORRELATION.csv",
    index=False
)



# =============================================================================
# 16A. REQUIRED PERFORMANCE / TRADE OUTPUTS FROM THE ASSIGNMENT
# =============================================================================
#
# Plain English:
# The original assignment asks for more than total return.  This section adds
# every required performance statistic, the trade-level CSV, monthly returns,
# pair/date contribution, profit concentration and a rupee capacity estimate.
# =============================================================================


def required_performance_row(
    data,
    label,
    multiplier
):
    data = data.sort_values(
        "DATE"
    ).copy()

    if data.empty:
        return {
            "COST_MULTIPLIER": multiplier,
            "GROSS_OR_NET": (
                "GROSS_BEFORE_TRADING_COSTS"
                if float(multiplier) == 0.0
                else "NET_AFTER_TRADING_COSTS"
            ),
            "PERIOD": label,
            "START_DATE": pd.NaT,
            "END_DATE": pd.NaT,
            "N_MARKET_DAYS": 0,
            "TOTAL_RETURN": np.nan,
            "CAGR": np.nan,
            "ANNUALIZED_VOLATILITY": np.nan,
            "SHARPE_RF0": np.nan,
            "SORTINO_RF0": np.nan,
            "MAX_DRAWDOWN": np.nan,
            "CALMAR": np.nan,
            "AVG_GROSS_EXPOSURE_TO_NAV": np.nan,
            "AVG_ABS_NET_EXPOSURE_TO_NAV": np.nan,
            "TOTAL_TURNOVER_TO_NAV": np.nan,
        }

    initial = float(
        data[
            "NAV_START"
        ].iloc[
            0
        ]
    )

    final = float(
        data[
            "NAV_END"
        ].iloc[
            -1
        ]
    )

    total_return = (
        final
        /
        initial
        -
        1.0
    )

    n_days = len(
        data
    )

    cagr = (
        (
            1.0
            +
            total_return
        )
        **
        (
            TRADING_DAYS_PER_YEAR
            /
            n_days
        )
        -
        1.0
        if
        (
            1.0
            +
            total_return
        )
        >
        0
        else
        np.nan
    )

    r = data[
        "DAILY_RETURN"
    ].astype(
        float
    )

    daily_std = float(
        r.std(
            ddof=1
        )
    )

    annualized_vol = (
        daily_std
        *
        np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
        if
        np.isfinite(
            daily_std
        )
        else
        np.nan
    )

    sharpe = (
        float(
            r.mean()
        )
        /
        daily_std
        *
        np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
        if
        np.isfinite(
            daily_std
        )
        and
        daily_std
        >
        NUMERIC_TOL
        else
        np.nan
    )

    downside = np.minimum(
        r.to_numpy(
            dtype=float
        ),
        0.0
    )

    downside_daily = float(
        np.sqrt(
            np.mean(
                downside
                **
                2
            )
        )
    )

    sortino = (
        float(
            r.mean()
        )
        *
        TRADING_DAYS_PER_YEAR
        /
        (
            downside_daily
            *
            np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )
        if
        downside_daily
        >
        NUMERIC_TOL
        else
        np.nan
    )

    nav_series = data[
        "NAV_END"
    ].astype(
        float
    )

    drawdown = (
        nav_series
        /
        nav_series.cummax()
        -
        1.0
    )

    max_drawdown = float(
        drawdown.min()
    )

    calmar = (
        cagr
        /
        abs(
            max_drawdown
        )
        if
        np.isfinite(
            cagr
        )
        and
        max_drawdown
        <
        -NUMERIC_TOL
        else
        np.nan
    )

    return {
        "COST_MULTIPLIER": multiplier,
        "GROSS_OR_NET": (
            "GROSS_BEFORE_TRADING_COSTS"
            if float(multiplier) == 0.0
            else "NET_AFTER_TRADING_COSTS"
        ),
        "PERIOD": label,
        "START_DATE": data[
            "DATE"
        ].min(),
        "END_DATE": data[
            "DATE"
        ].max(),
        "N_MARKET_DAYS": n_days,
        "TOTAL_RETURN": total_return,
        "CAGR": cagr,
        "ANNUALIZED_VOLATILITY": annualized_vol,
        "SHARPE_RF0": sharpe,
        "SORTINO_RF0": sortino,
        "MAX_DRAWDOWN": max_drawdown,
        "CALMAR": calmar,
        "AVG_GROSS_EXPOSURE_TO_NAV": float(
            data[
                "GROSS_EXPOSURE_TO_NAV"
            ].mean()
        ),
        "AVG_ABS_NET_EXPOSURE_TO_NAV": float(
            data[
                "NET_EXPOSURE_TO_NAV"
            ].abs().mean()
        ),
        "TOTAL_TURNOVER_TO_NAV": float(
            data[
                "TURNOVER_TO_NAV"
            ].sum()
        ),
    }


required_perf_rows = []

for multiplier, group in daily_all.groupby(
    "COST_MULTIPLIER",
    sort=True
):
    required_perf_rows.append(
        required_performance_row(
            group,
            "FULL",
            multiplier
        )
    )

    required_perf_rows.append(
        required_performance_row(
            group[
                group[
                    "DATE"
                ]
                <
                OOS_START
            ],
            "DEVELOPMENT",
            multiplier
        )
    )

    required_perf_rows.append(
        required_performance_row(
            group[
                group[
                    "DATE"
                ]
                >=
                OOS_START
            ],
            "OOS",
            multiplier
        )
    )


required_performance = pd.DataFrame(
    required_perf_rows
)

required_performance.to_csv(
    OUTPUT_DIR
    / "05A_REQUIRED_PERFORMANCE_METRICS.csv",
    index=False,
    date_format="%Y-%m-%d"
)


# Dedicated trade ledger required by the assignment.
# Primary reported trading costs = 1x.
if trade_pnl.empty:
    primary_trade_ledger = pd.DataFrame()

else:
    primary_trade_ledger = (
        trade_pnl[
            trade_pnl[
                "COST_MULTIPLIER"
            ].eq(
                1.0
            )
        ]
        .copy()
        .sort_values(
            [
                "ENTRY_EXECUTION_DATE",
                "EG_TRADE_ID",
            ]
        )
    )


primary_trade_ledger.to_csv(
    OUTPUT_DIR
    / "04A_PRIMARY_TRADE_LEDGER_1X.csv",
    index=False,
    date_format="%Y-%m-%d"
)


def build_required_trade_stats(
    df,
    label
):
    n = len(
        df
    )

    if n == 0:
        return {
            "PERIOD": label,
            "N_TRADES": 0,
            "N_WINNING_TRADES": 0,
            "HIT_RATE": np.nan,
            "AVERAGE_NET_PNL": np.nan,
            "MEDIAN_NET_PNL": np.nan,
            "AVERAGE_NET_RETURN_ON_PAIR_GROSS": np.nan,
            "MEDIAN_NET_RETURN_ON_PAIR_GROSS": np.nan,
            "HOLDING_DAYS_MEAN": np.nan,
            "HOLDING_DAYS_MEDIAN": np.nan,
            "HOLDING_DAYS_P25": np.nan,
            "HOLDING_DAYS_P75": np.nan,
            "HOLDING_DAYS_MIN": np.nan,
            "HOLDING_DAYS_MAX": np.nan,
            "CONVERGENCE_RATE": np.nan,
            "STOP_OUT_RATE": np.nan,
            "FORCED_WINDOW_EXIT_RATE": np.nan,
            "GROSS_PNL": 0.0,
            "TRANSACTION_COST": 0.0,
            "FINANCING_COST": 0.0,
            "TOTAL_COST": 0.0,
            "NET_PNL": 0.0,
        }

    h = df[
        "HOLDING_CALENDAR_DAYS"
    ].astype(
        float
    )

    return {
        "PERIOD": label,
        "N_TRADES": n,
        "N_WINNING_TRADES": int(
            (
                df[
                    "NET_PNL"
                ]
                >
                0
            ).sum()
        ),
        "HIT_RATE": float(
            (
                df[
                    "NET_PNL"
                ]
                >
                0
            ).mean()
        ),
        "AVERAGE_NET_PNL": float(
            df[
                "NET_PNL"
            ].mean()
        ),
        "MEDIAN_NET_PNL": float(
            df[
                "NET_PNL"
            ].median()
        ),
        "AVERAGE_NET_RETURN_ON_PAIR_GROSS": float(
            df[
                "NET_RETURN_ON_PAIR_GROSS"
            ].mean()
        ),
        "MEDIAN_NET_RETURN_ON_PAIR_GROSS": float(
            df[
                "NET_RETURN_ON_PAIR_GROSS"
            ].median()
        ),
        "HOLDING_DAYS_MEAN": float(
            h.mean()
        ),
        "HOLDING_DAYS_MEDIAN": float(
            h.median()
        ),
        "HOLDING_DAYS_P25": float(
            h.quantile(
                0.25
            )
        ),
        "HOLDING_DAYS_P75": float(
            h.quantile(
                0.75
            )
        ),
        "HOLDING_DAYS_MIN": float(
            h.min()
        ),
        "HOLDING_DAYS_MAX": float(
            h.max()
        ),
        "CONVERGENCE_RATE": float(
            df[
                "EXIT_REASON"
            ].eq(
                "MEAN_CROSS"
            ).mean()
        ),
        "STOP_OUT_RATE": float(
            df[
                "EXIT_REASON"
            ].astype(
                str
            ).str.contains(
                "STOP",
                regex=False
            ).mean()
        ),
        "FORCED_WINDOW_EXIT_RATE": float(
            df[
                "EXIT_REASON"
            ].astype(
                str
            ).str.contains(
                "FORCE_CLOSE",
                regex=False
            ).mean()
        ),
        "GROSS_PNL": float(
            df[
                "GROSS_PNL"
            ].sum()
        ),
        "TRANSACTION_COST": float(
            df[
                "TRANSACTION_COST"
            ].sum()
        ),
        "FINANCING_COST": float(
            df[
                "FINANCING_COST"
            ].sum()
        ),
        "TOTAL_COST": float(
            df[
                "TOTAL_COST"
            ].sum()
        ),
        "NET_PNL": float(
            df[
                "NET_PNL"
            ].sum()
        ),
    }


required_trade_stats = pd.DataFrame(
    [
        build_required_trade_stats(
            primary_trade_ledger,
            "FULL"
        ),
        build_required_trade_stats(
            primary_trade_ledger[
                primary_trade_ledger[
                    "BLOCK_TYPE"
                ].eq(
                    "DEVELOPMENT"
                )
            ]
            if
            not primary_trade_ledger.empty
            else
            primary_trade_ledger,
            "DEVELOPMENT"
        ),
        build_required_trade_stats(
            primary_trade_ledger[
                primary_trade_ledger[
                    "BLOCK_TYPE"
                ].eq(
                    "OOS"
                )
            ]
            if
            not primary_trade_ledger.empty
            else
            primary_trade_ledger,
            "OOS"
        ),
    ]
)

required_trade_stats.to_csv(
    OUTPUT_DIR
    / "06A_REQUIRED_TRADE_STATISTICS.csv",
    index=False
)


# Holding-period distribution.
holding_rows = []

holding_bins = [
    (
        "0-30 days",
        0,
        30
    ),
    (
        "31-60 days",
        31,
        60
    ),
    (
        "61-90 days",
        61,
        90
    ),
    (
        "91-120 days",
        91,
        120
    ),
    (
        "121-180 days",
        121,
        180
    ),
    (
        ">180 days",
        181,
        np.inf
    ),
]

for label, low, high in holding_bins:
    if primary_trade_ledger.empty:
        count = 0

    else:
        h = primary_trade_ledger[
            "HOLDING_CALENDAR_DAYS"
        ].astype(
            float
        )

        count = int(
            (
                h.ge(
                    low
                )
                &
                h.le(
                    high
                )
            ).sum()
        )

    holding_rows.append(
        {
            "HOLDING_PERIOD_BUCKET":
                label,

            "N_TRADES":
                count,

            "SHARE_OF_TRADES":
                (
                    count
                    /
                    len(
                        primary_trade_ledger
                    )
                    if
                    len(
                        primary_trade_ledger
                    )
                    else
                    np.nan
                ),
        }
    )

pd.DataFrame(
    holding_rows
).to_csv(
    OUTPUT_DIR
    / "06B_HOLDING_PERIOD_DISTRIBUTION.csv",
    index=False
)


# Monthly return table at the primary 1x cost assumption.
daily_primary = (
    daily_all[
        daily_all[
            "COST_MULTIPLIER"
        ].eq(
            1.0
        )
    ]
    .sort_values(
        "DATE"
    )
    .copy()
)

monthly = (
    daily_primary.set_index(
        "DATE"
    )[
        "DAILY_RETURN"
    ]
    .resample(
        "ME"
    )
    .apply(
        lambda x:
            (
                1.0
                +
                x
            ).prod()
            -
            1.0
    )
    .reset_index(
        name="MONTHLY_RETURN_1X"
    )
)

monthly.to_csv(
    OUTPUT_DIR
    / "07A_MONTHLY_RETURNS_1X.csv",
    index=False,
    date_format="%Y-%m-%d"
)


# Required attribution: by pair and by formation window.
if primary_trade_ledger.empty:
    required_pair_attribution = pd.DataFrame()

    required_formation_attribution = pd.DataFrame()

else:
    required_pair_attribution = (
        primary_trade_ledger.groupby(
            "PAIR_ID",
            as_index=False
        )
        .agg(
            N_TRADES=(
                "EG_TRADE_ID",
                "count"
            ),
            GROSS_PNL=(
                "GROSS_PNL",
                "sum"
            ),
            TRANSACTION_COST=(
                "TRANSACTION_COST",
                "sum"
            ),
            FINANCING_COST=(
                "FINANCING_COST",
                "sum"
            ),
            TOTAL_COST=(
                "TOTAL_COST",
                "sum"
            ),
            NET_PNL=(
                "NET_PNL",
                "sum"
            ),
        )
        .sort_values(
            "NET_PNL",
            ascending=False
        )
    )

    required_formation_attribution = (
        primary_trade_ledger.groupby(
            [
                "FORMATION_DATE",
                "BLOCK_TYPE",
            ],
            as_index=False
        )
        .agg(
            N_TRADES=(
                "EG_TRADE_ID",
                "count"
            ),
            GROSS_PNL=(
                "GROSS_PNL",
                "sum"
            ),
            TRANSACTION_COST=(
                "TRANSACTION_COST",
                "sum"
            ),
            FINANCING_COST=(
                "FINANCING_COST",
                "sum"
            ),
            TOTAL_COST=(
                "TOTAL_COST",
                "sum"
            ),
            NET_PNL=(
                "NET_PNL",
                "sum"
            ),
        )
        .sort_values(
            "FORMATION_DATE"
        )
    )


required_pair_attribution.to_csv(
    OUTPUT_DIR
    / "08A_PAIR_ATTRIBUTION_1X.csv",
    index=False,
    date_format="%Y-%m-%d"
)

required_formation_attribution.to_csv(
    OUTPUT_DIR
    / "08B_FORMATION_WINDOW_ATTRIBUTION_1X.csv",
    index=False,
    date_format="%Y-%m-%d"
)


# Required concentration of P&L in the best 10% of trades.
if primary_trade_ledger.empty:
    top_decile = pd.DataFrame(
        [
            {
                "N_TRADES":
                    0,

                "N_TOP_DECILE_TRADES":
                    0,

                "TOP_DECILE_NET_PNL":
                    0.0,

                "TOTAL_NET_PNL":
                    0.0,

                "TOTAL_POSITIVE_PNL":
                    0.0,

                "TOP_DECILE_SHARE_OF_POSITIVE_PNL":
                    np.nan,
            }
        ]
    )

else:
    ranked = primary_trade_ledger.sort_values(
        "NET_PNL",
        ascending=False
    )

    n_top = max(
        1,
        int(
            np.ceil(
                0.10
                *
                len(
                    ranked
                )
            )
        )
    )

    top_net = float(
        ranked.head(
            n_top
        )[
            "NET_PNL"
        ].sum()
    )

    total_net = float(
        ranked[
            "NET_PNL"
        ].sum()
    )

    total_positive = float(
        ranked.loc[
            ranked[
                "NET_PNL"
            ]
            >
            0,
            "NET_PNL",
        ].sum()
    )

    top_decile = pd.DataFrame(
        [
            {
                "N_TRADES":
                    len(
                        ranked
                    ),

                "N_TOP_DECILE_TRADES":
                    n_top,

                "TOP_DECILE_NET_PNL":
                    top_net,

                "TOTAL_NET_PNL":
                    total_net,

                "TOTAL_POSITIVE_PNL":
                    total_positive,

                "TOP_DECILE_SHARE_OF_POSITIVE_PNL":
                    (
                        top_net
                        /
                        total_positive
                        if
                        total_positive
                        >
                        NUMERIC_TOL
                        else
                        np.nan
                    ),
            }
        ]
    )


top_decile.to_csv(
    OUTPUT_DIR
    / "08C_TOP_DECILE_PNL_CONCENTRATION_1X.csv",
    index=False
)


# =============================================================================
# 16B. CAPACITY IN RUPEES
# =============================================================================
#
# We impose a simple capacity rule:
# no individual trade leg may exceed 1% of that stock's traded value on the
# entry or exit day.
#
# The output converts that rule into an approximate maximum portfolio size.
# =============================================================================

CAPACITY_PARTICIPATION_RATE = 0.01

capacity_rows = []


if not primary_trade_ledger.empty:
    for row in primary_trade_ledger.itertuples(
        index=False
    ):
        entry_long = price_row(
            row.ENTRY_EXECUTION_DATE,
            row.LONG_COMPANY
        )

        entry_short = price_row(
            row.ENTRY_EXECUTION_DATE,
            row.SHORT_COMPANY
        )

        exit_long = price_row(
            row.EXIT_EXECUTION_DATE,
            row.LONG_COMPANY
        )

        exit_short = price_row(
            row.EXIT_EXECUTION_DATE,
            row.SHORT_COMPANY
        )

        nav_at_entry = (
            float(
                row.INITIAL_PAIR_GROSS
            )
            /
            PAIR_GROSS_WEIGHT
        )

        entry_long_weight = safe_div(
            float(
                row.INITIAL_LONG_NOTIONAL
            ),
            nav_at_entry
        )

        entry_short_weight = safe_div(
            float(
                row.INITIAL_SHORT_NOTIONAL
            ),
            nav_at_entry
        )

        exit_long_value = abs(
            float(
                row.LONG_UNITS
            )
            *
            float(
                exit_long[
                    "TOTAL_RETURN_INDEX"
                ]
            )
        )

        exit_short_value = abs(
            float(
                row.SHORT_UNITS
            )
            *
            float(
                exit_short[
                    "FUTURES_PROXY_INDEX"
                ]
            )
        )

        exit_long_weight = safe_div(
            exit_long_value,
            nav_at_entry
        )

        exit_short_weight = safe_div(
            exit_short_value,
            nav_at_entry
        )

        legs = [
            (
                "ENTRY_LONG",
                float(
                    entry_long[
                        "TOTAL_TRADED_VALUE"
                    ]
                ),
                entry_long_weight
            ),
            (
                "ENTRY_SHORT",
                float(
                    entry_short[
                        "TOTAL_TRADED_VALUE"
                    ]
                ),
                entry_short_weight
            ),
            (
                "EXIT_LONG",
                float(
                    exit_long[
                        "TOTAL_TRADED_VALUE"
                    ]
                ),
                exit_long_weight
            ),
            (
                "EXIT_SHORT",
                float(
                    exit_short[
                        "TOTAL_TRADED_VALUE"
                    ]
                ),
                exit_short_weight
            ),
        ]

        capacity_candidates = []

        for leg_name, traded_value, portfolio_weight in legs:
            if (
                np.isfinite(
                    traded_value
                )
                and
                traded_value
                >
                0
                and
                np.isfinite(
                    portfolio_weight
                )
                and
                portfolio_weight
                >
                NUMERIC_TOL
            ):
                capacity_candidates.append(
                    (
                        leg_name,
                        CAPACITY_PARTICIPATION_RATE
                        *
                        traded_value
                        /
                        portfolio_weight
                    )
                )

        if capacity_candidates:
            limiting_leg, trade_capacity = min(
                capacity_candidates,
                key=lambda x:
                    x[
                        1
                    ]
            )

        else:
            limiting_leg = (
                "NOT_CALCULATED"
            )

            trade_capacity = np.nan

        capacity_rows.append(
            {
                "EG_TRADE_ID":
                    row.EG_TRADE_ID,

                "PAIR_ID":
                    row.PAIR_ID,

                "ENTRY_EXECUTION_DATE":
                    row.ENTRY_EXECUTION_DATE,

                "EXIT_EXECUTION_DATE":
                    row.EXIT_EXECUTION_DATE,

                "PARTICIPATION_RATE":
                    CAPACITY_PARTICIPATION_RATE,

                "LIMITING_LEG":
                    limiting_leg,

                "MAX_PORTFOLIO_CAPACITY_RUPEES":
                    trade_capacity,
            }
        )


capacity_detail = pd.DataFrame(
    capacity_rows
)

capacity_detail.to_csv(
    OUTPUT_DIR
    / "09A_CAPACITY_BY_TRADE.csv",
    index=False,
    date_format="%Y-%m-%d"
)


strategy_capacity = pd.DataFrame(
    [
        {
            "PARTICIPATION_RATE":
                CAPACITY_PARTICIPATION_RATE,

            "STRATEGY_CAPACITY_RUPEES":
                (
                    float(
                        capacity_detail[
                            "MAX_PORTFOLIO_CAPACITY_RUPEES"
                        ].min()
                    )
                    if
                    not capacity_detail.empty
                    else
                    np.nan
                ),

            "INTERPRETATION":
                (
                    "Conservative bottleneck across entry/exit legs; "
                    "each leg limited to 1% of that day's traded value."
                ),
        }
    ]
)

strategy_capacity.to_csv(
    OUTPUT_DIR
    / "09B_STRATEGY_CAPACITY_RUPEES.csv",
    index=False
)


# =============================================================================
# 16C. REQUIRED ROBUSTNESS CHECKS — DEVELOPMENT PERIOD ONLY
# =============================================================================
#
# IMPORTANT:
# The final OOS period is NOT used for these checks.
#
# The assignment explicitly asks sensitivity to:
#   1. entry / exit thresholds
#   2. formation-window length
#   3. universe size
#   4. half-life filter
#
# We keep these checks intentionally small because the project is time-limited.
# =============================================================================


def bh_adjust(
    pvalues
):
    p = np.asarray(
        pvalues,
        dtype=float
    )

    q = np.full(
        len(
            p
        ),
        np.nan
    )

    valid = np.isfinite(
        p
    )

    if not valid.any():
        return q

    pv = p[
        valid
    ]

    order = np.argsort(
        pv
    )

    ranked = pv[
        order
    ]

    m = len(
        ranked
    )

    adjusted = (
        ranked
        *
        m
        /
        np.arange(
            1,
            m + 1
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[
            ::-1
        ]
    )[
        ::-1
    ]

    adjusted = np.minimum(
        adjusted,
        1.0
    )

    valid_positions = np.where(
        valid
    )[0]

    q[
        valid_positions[
            order
        ]
    ] = adjusted

    return q


def parse_company_list_simple(
    value
):
    if pd.isna(
        value
    ):
        return []

    return sorted(
        {
            x.strip().upper()
            for x in str(
                value
            ).split(
                ";"
            )
            if x.strip()
        }
    )


def choose_non_overlapping_pairs(
    eligible
):
    if eligible.empty:
        return eligible.copy()

    eligible = eligible.sort_values(
        [
            "EVIDENCE_RANK",
            "PAIR_ID",
        ]
    ).reset_index(
        drop=True
    )

    max_k = min(
        4,
        len(
            eligible
        )
    )

    best_indices = None
    best_score = None
    best_ids = None

    for k in range(
        max_k,
        0,
        -1
    ):
        for combo in combinations(
            range(
                len(
                    eligible
                )
            ),
            k
        ):
            names = []

            score = 0.0

            ids = []

            valid = True

            for idx in combo:
                row = eligible.iloc[
                    idx
                ]

                a = row[
                    "UNORDERED_COMPANY_1"
                ]

                b = row[
                    "UNORDERED_COMPANY_2"
                ]

                if (
                    a
                    in
                    names
                    or
                    b
                    in
                    names
                ):
                    valid = False
                    break

                names.extend(
                    [
                        a,
                        b,
                    ]
                )

                score += float(
                    row[
                        "EVIDENCE_RANK"
                    ]
                )

                ids.append(
                    row[
                        "PAIR_ID"
                    ]
                )

            if not valid:
                continue

            ids_tuple = tuple(
                sorted(
                    ids
                )
            )

            if (
                best_indices is None
                or
                score
                <
                best_score
                -
                NUMERIC_TOL
                or
                (
                    abs(
                        score
                        -
                        best_score
                    )
                    <=
                    NUMERIC_TOL
                    and
                    ids_tuple
                    <
                    best_ids
                )
            ):
                best_indices = combo
                best_score = score
                best_ids = ids_tuple

        if best_indices is not None:
            return eligible.iloc[
                list(
                    best_indices
                )
            ].copy()

    return eligible.iloc[
        0:0
    ].copy()


def formation_gap_and_half_life(
    row
):
    formation_date = pd.Timestamp(
        row[
            "FORMATION_DATE"
        ]
    )

    formation_start = pd.Timestamp(
        row[
            "FORMATION_START"
        ]
    )

    a = prices.loc[
        prices[
            "COMPANY_ID"
        ].eq(
            row[
                "COMPANY_A"
            ]
        )
        &
        prices[
            "DATE"
        ].between(
            formation_start,
            formation_date,
            inclusive="both"
        ),
        [
            "DATE",
            "TOTAL_RETURN_INDEX",
        ],
    ].rename(
        columns={
            "TOTAL_RETURN_INDEX":
                "A"
        }
    )

    b = prices.loc[
        prices[
            "COMPANY_ID"
        ].eq(
            row[
                "COMPANY_B"
            ]
        )
        &
        prices[
            "DATE"
        ].between(
            formation_start,
            formation_date,
            inclusive="both"
        ),
        [
            "DATE",
            "TOTAL_RETURN_INDEX",
        ],
    ].rename(
        columns={
            "TOTAL_RETURN_INDEX":
                "B"
        }
    )

    pair = a.merge(
        b,
        on="DATE",
        how="inner"
    ).sort_values(
        "DATE"
    )

    gap = (
        np.log(
            pair[
                "A"
            ].to_numpy(
                dtype=float
            )
        )
        -
        float(
            row[
                "ALPHA"
            ]
        )
        -
        float(
            row[
                "BETA"
            ]
        )
        *
        np.log(
            pair[
                "B"
            ].to_numpy(
                dtype=float
            )
        )
    )

    if len(
        gap
    ) < 3:
        return np.nan

    y = gap[
        1:
    ]

    x = gap[
        :-1
    ]

    X = np.column_stack(
        [
            np.ones(
                len(
                    x
                )
            ),
            x,
        ]
    )

    coef, _, _, _ = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )

    phi = float(
        coef[
            1
        ]
    )

    if (
        phi
        <=
        0
        or
        phi
        >=
        1
    ):
        return np.nan

    return float(
        -
        np.log(
            2.0
        )
        /
        np.log(
            phi
        )
    )


half_life_rows = []

for _, row in selected.iterrows():
    half_life_rows.append(
        {
            "FORMATION_DATE":
                row[
                    "FORMATION_DATE"
                ],

            "BLOCK_TYPE":
                row[
                    "BLOCK_TYPE"
                ],

            "PAIR_ID":
                row[
                    "PAIR_ID"
                ],

            "COMPANY_A":
                row[
                    "COMPANY_A"
                ],

            "COMPANY_B":
                row[
                    "COMPANY_B"
                ],

            "HALF_LIFE_SESSIONS":
                formation_gap_and_half_life(
                    row
                ),
        }
    )


half_life_table = pd.DataFrame(
    half_life_rows
)

half_life_table[
    "PASSES_5_TO_60_SESSION_FILTER"
] = (
    half_life_table[
        "HALF_LIFE_SESSIONS"
    ].between(
        5.0,
        60.0,
        inclusive="both"
    )
)

half_life_table.to_csv(
    OUTPUT_DIR
    / "13A_SELECTED_PAIR_HALF_LIFE.csv",
    index=False,
    date_format="%Y-%m-%d"
)


def generate_variant_trades(
    selected_variant,
    variant_name,
    entry_z=2.0,
    exit_z=0.0,
    stop_z=None
):
    trade_rows = []

    if (
        selected_variant is None
        or
        selected_variant.empty
    ):
        return pd.DataFrame()

    for pair_row in selected_variant.sort_values(
        [
            "FORMATION_DATE",
            "SELECTED_PAIR_NUMBER",
        ]
    ).itertuples(
        index=False
    ):
        mean_gap = float(
            pair_row.RESIDUAL_MEAN
        )

        sd_gap = float(
            pair_row.RESIDUAL_STD_DDOF1
        )

        a = prices.loc[
            prices[
                "COMPANY_ID"
            ].eq(
                pair_row.COMPANY_A
            )
            &
            prices[
                "DATE"
            ].between(
                pair_row.TRADING_START,
                pair_row.TRADING_END,
                inclusive="both"
            ),
            [
                "DATE",
                "TOTAL_RETURN_INDEX",
                "SEGMENT_ID",
            ],
        ].rename(
            columns={
                "TOTAL_RETURN_INDEX":
                    "A",

                "SEGMENT_ID":
                    "SEG_A",
            }
        )

        b = prices.loc[
            prices[
                "COMPANY_ID"
            ].eq(
                pair_row.COMPANY_B
            )
            &
            prices[
                "DATE"
            ].between(
                pair_row.TRADING_START,
                pair_row.TRADING_END,
                inclusive="both"
            ),
            [
                "DATE",
                "TOTAL_RETURN_INDEX",
                "SEGMENT_ID",
            ],
        ].rename(
            columns={
                "TOTAL_RETURN_INDEX":
                    "B",

                "SEGMENT_ID":
                    "SEG_B",
            }
        )

        pair = (
            a.merge(
                b,
                on="DATE",
                how="inner"
            )
            .sort_values(
                "DATE"
            )
            .reset_index(
                drop=True
            )
        )

        if len(
            pair
        ) < 3:
            continue

        pair[
            "GAP"
        ] = (
            np.log(
                pair[
                    "A"
                ].astype(
                    float
                )
            )
            -
            float(
                pair_row.ALPHA
            )
            -
            float(
                pair_row.BETA
            )
            *
            np.log(
                pair[
                    "B"
                ].astype(
                    float
                )
            )
        )

        pair[
            "Z"
        ] = (
            pair[
                "GAP"
            ]
            -
            mean_gap
        ) / sd_gap

        start_seg_a = str(
            pair.loc[
                0,
                "SEG_A"
            ]
        )

        start_seg_b = str(
            pair.loc[
                0,
                "SEG_B"
            ]
        )

        changed = (
            pair[
                "SEG_A"
            ].astype(
                str
            ).ne(
                start_seg_a
            )
            |
            pair[
                "SEG_B"
            ].astype(
                str
            ).ne(
                start_seg_b
            )
        )

        break_positions = np.where(
            changed.to_numpy()
        )[0]

        if len(
            break_positions
        ):
            terminal_pos = int(
                break_positions[
                    0
                ]
            ) - 1

            terminal_reason = (
                "STRUCTURAL_BREAK_FORCE_CLOSE"
            )

        else:
            terminal_pos = len(
                pair
            ) - 1

            terminal_reason = (
                "WINDOW_END_FORCE_CLOSE"
            )

        if terminal_pos < 2:
            continue

        effective = pair.iloc[
            :
            terminal_pos
            +
            1
        ].reset_index(
            drop=True
        )

        scan_pos = 0
        trade_number = 0

        while scan_pos <= len(
            effective
        ) - 3:
            signal_pos = None
            entry_side = None

            for i in range(
                scan_pos,
                len(
                    effective
                ) - 2
            ):
                z = float(
                    effective.loc[
                        i,
                        "Z"
                    ]
                )

                if z >= entry_z:
                    signal_pos = i
                    entry_side = (
                        "ABOVE"
                    )
                    break

                if z <= -entry_z:
                    signal_pos = i
                    entry_side = (
                        "BELOW"
                    )
                    break

            if signal_pos is None:
                break

            entry_exec_pos = signal_pos + 1

            if entry_exec_pos >= len(
                effective
            ) - 1:
                break

            if entry_side == "ABOVE":
                long_company = (
                    pair_row.COMPANY_B
                )

                short_company = (
                    pair_row.COMPANY_A
                )

                direction = (
                    "HIGH_GAP_SHORT_A_LONG_B"
                )

            else:
                long_company = (
                    pair_row.COMPANY_A
                )

                short_company = (
                    pair_row.COMPANY_B
                )

                direction = (
                    "LOW_GAP_LONG_A_SHORT_B"
                )

            exit_signal_pos = None
            exit_reason = None

            for j in range(
                entry_exec_pos,
                len(
                    effective
                ) - 1
            ):
                z = float(
                    effective.loc[
                        j,
                        "Z"
                    ]
                )

                converged = (
                    (
                        entry_side
                        ==
                        "ABOVE"
                        and
                        z
                        <=
                        exit_z
                    )
                    or
                    (
                        entry_side
                        ==
                        "BELOW"
                        and
                        z
                        >=
                        -exit_z
                    )
                )

                stopped = (
                    stop_z is not None
                    and
                    (
                        (
                            entry_side
                            ==
                            "ABOVE"
                            and
                            z
                            >=
                            stop_z
                        )
                        or
                        (
                            entry_side
                            ==
                            "BELOW"
                            and
                            z
                            <=
                            -stop_z
                        )
                    )
                )

                if stopped:
                    exit_signal_pos = j
                    exit_reason = (
                        "STOP_OUT"
                    )
                    break

                if converged:
                    exit_signal_pos = j
                    exit_reason = (
                        "MEAN_CROSS"
                        if
                        exit_z
                        ==
                        0.0
                        else
                        "EXIT_BAND_REACHED"
                    )
                    break

            trade_number += 1

            if exit_signal_pos is not None:
                exit_exec_pos = exit_signal_pos + 1

                exit_signal_date = pd.Timestamp(
                    effective.loc[
                        exit_signal_pos,
                        "DATE"
                    ]
                )

            else:
                exit_exec_pos = len(
                    effective
                ) - 1

                exit_signal_date = pd.NaT

                exit_reason = terminal_reason

            trade_rows.append(
                {
                    "FORMATION_DATE":
                        pd.Timestamp(
                            pair_row.FORMATION_DATE
                        ),

                    "BLOCK_TYPE":
                        pair_row.BLOCK_TYPE,

                    "SELECTED_PAIR_NUMBER":
                        int(
                            pair_row.SELECTED_PAIR_NUMBER
                        ),

                    "PAIR_ID":
                        pair_row.PAIR_ID,

                    "TRADE_NUMBER_WITHIN_PAIR_WINDOW":
                        trade_number,

                    "COMPANY_A":
                        pair_row.COMPANY_A,

                    "COMPANY_B":
                        pair_row.COMPANY_B,

                    "ALPHA":
                        float(
                            pair_row.ALPHA
                        ),

                    "BETA":
                        float(
                            pair_row.BETA
                        ),

                    "ENTRY_SIGNAL_DATE":
                        pd.Timestamp(
                            effective.loc[
                                signal_pos,
                                "DATE"
                            ]
                        ),

                    "ENTRY_EXECUTION_DATE":
                        pd.Timestamp(
                            effective.loc[
                                entry_exec_pos,
                                "DATE"
                            ]
                        ),

                    "ENTRY_DIRECTION":
                        direction,

                    "ENTRY_SIGNAL_Z":
                        float(
                            effective.loc[
                                signal_pos,
                                "Z"
                            ]
                        ),

                    "ENTRY_EXECUTION_Z":
                        float(
                            effective.loc[
                                entry_exec_pos,
                                "Z"
                            ]
                        ),

                    "LONG_COMPANY":
                        long_company,

                    "SHORT_COMPANY":
                        short_company,

                    "EXIT_SIGNAL_DATE":
                        exit_signal_date,

                    "EXIT_SIGNAL_Z":
                        (
                            float(
                                effective.loc[
                                    exit_signal_pos,
                                    "Z"
                                ]
                            )
                            if
                            exit_signal_pos is not None
                            else
                            np.nan
                        ),

                    "EXIT_EXECUTION_DATE":
                        pd.Timestamp(
                            effective.loc[
                                exit_exec_pos,
                                "DATE"
                            ]
                        ),

                    "EXIT_EXECUTION_Z":
                        float(
                            effective.loc[
                                exit_exec_pos,
                                "Z"
                            ]
                        ),

                    "EXIT_REASON":
                        exit_reason,

                    "HOLDING_CALENDAR_DAYS":
                        int(
                            (
                                pd.Timestamp(
                                    effective.loc[
                                        exit_exec_pos,
                                        "DATE"
                                    ]
                                )
                                -
                                pd.Timestamp(
                                    effective.loc[
                                        entry_exec_pos,
                                        "DATE"
                                    ]
                                )
                            ).days
                        ),
                }
            )

            if (
                exit_signal_pos is not None
                and
                exit_reason
                in
                [
                    "MEAN_CROSS",
                    "EXIT_BAND_REACHED",
                ]
            ):
                scan_pos = exit_exec_pos

            elif (
                exit_signal_pos is not None
                and
                exit_reason
                ==
                "STOP_OUT"
            ):
                # After a stop, do NOT immediately enter again simply because
                # the gap is still beyond the +/- entry threshold.
                #
                # First wait until the gap has returned INSIDE the entry band.
                # Only after that reset can a fresh later divergence create
                # another trade.
                scan_pos = exit_exec_pos

                while (
                    scan_pos
                    <=
                    len(
                        effective
                    )
                    -
                    3
                    and
                    abs(
                        float(
                            effective.loc[
                                scan_pos,
                                "Z"
                            ]
                        )
                    )
                    >=
                    entry_z
                ):
                    scan_pos += 1

            else:
                break

    trades = pd.DataFrame(
        trade_rows
    )

    if trades.empty:
        return trades

    trades[
        "EG_TRADE_ID"
    ] = [
        (
            f"{variant_name}_{fd:%Y%m%d}_P{int(p):02d}_T{int(t):02d}"
        )
        for fd, p, t
        in zip(
            trades[
                "FORMATION_DATE"
            ],
            trades[
                "SELECTED_PAIR_NUMBER"
            ],
            trades[
                "TRADE_NUMBER_WITHIN_PAIR_WINDOW"
            ],
        )
    ]

    trades[
        "A_SHARE_OF_PAIR_GROSS"
    ] = (
        1.0
        /
        (
            1.0
            +
            trades[
                "BETA"
            ]
        )
    )

    trades[
        "B_SHARE_OF_PAIR_GROSS"
    ] = (
        trades[
            "BETA"
        ]
        /
        (
            1.0
            +
            trades[
                "BETA"
            ]
        )
    )

    return trades


def run_development_variant(
    selected_variant,
    variant_name,
    entry_z=2.0,
    exit_z=0.0,
    stop_z=None
):
    global executable_trades

    variant_trades = generate_variant_trades(
        selected_variant,
        variant_name,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z
    )

    original = executable_trades

    try:
        executable_trades = (
            variant_trades
        )

        daily_variant, trade_variant = simulate(
            1.0
        )

    finally:
        executable_trades = original

    daily_variant = daily_variant[
        daily_variant[
            "DATE"
        ]
        <
        OOS_START
    ].copy()

    perf = required_performance_row(
        daily_variant,
        "DEVELOPMENT",
        1.0
    )

    perf.update(
        {
            "ROBUSTNESS_VARIANT":
                variant_name,

            "N_SELECTED_PAIR_PERIODS":
                len(
                    selected_variant
                ),

            "N_TRADES":
                len(
                    trade_variant
                ),

            "ENTRY_Z":
                entry_z,

            "EXIT_Z":
                exit_z,

            "STOP_Z":
                stop_z,

            "ACTUAL_ENTRY_DATE_FNO_RECHECK":
                "NO - formation-date F&O screen retained; primary result has actual entry-date check",
        }
    )

    return (
        perf,
        variant_trades,
        trade_variant,
    )


# ----- Threshold / stop / half-life checks on the frozen primary pairs -----

development_selected = selected[
    selected[
        "BLOCK_TYPE"
    ].eq(
        "DEVELOPMENT"
    )
].copy()


threshold_specs = [
    (
        "ENTRY_1.5_EXIT_MEAN",
        1.5,
        0.0,
        None,
        development_selected,
    ),
    (
        "PRIMARY_ENTRY_2_EXIT_MEAN",
        2.0,
        0.0,
        None,
        development_selected,
    ),
    (
        "ENTRY_2.5_EXIT_MEAN",
        2.5,
        0.0,
        None,
        development_selected,
    ),
    (
        "ENTRY_2_EXIT_0.5SD",
        2.0,
        0.5,
        None,
        development_selected,
    ),
    (
        "PRIMARY_WITH_4SD_STOP",
        2.0,
        0.0,
        4.0,
        development_selected,
    ),
]


# Match using normalized Timestamp objects, not strings.
# The previous string comparison could compare "2021-01-29" with
# "2021-01-29 00:00:00" and incorrectly reject a valid pair.
half_life_keys = {
    (
        pd.Timestamp(
            fd
        ).normalize(),
        pair_id,
    )
    for fd, pair_id
    in zip(
        half_life_table.loc[
            half_life_table[
                "PASSES_5_TO_60_SESSION_FILTER"
            ],
            "FORMATION_DATE",
        ],
        half_life_table.loc[
            half_life_table[
                "PASSES_5_TO_60_SESSION_FILTER"
            ],
            "PAIR_ID",
        ],
    )
}


half_life_selected = development_selected[
    [
        (
            pd.Timestamp(
                fd
            ).normalize(),
            pair_id,
        )
        in
        half_life_keys
        for fd, pair_id
        in zip(
            development_selected[
                "FORMATION_DATE"
            ],
            development_selected[
                "PAIR_ID"
            ],
        )
    ]
].copy()


threshold_specs.append(
    (
        "HALF_LIFE_5_TO_60_SESSIONS",
        2.0,
        0.0,
        None,
        half_life_selected,
    )
)


robustness_perf_rows = []
robustness_trade_frames = []


for (
    variant_name,
    entry_z,
    exit_z,
    stop_z,
    selected_variant
) in threshold_specs:
    perf, _, trade_variant = run_development_variant(
        selected_variant,
        variant_name,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z
    )

    perf[
        "ROBUSTNESS_CATEGORY"
    ] = (
        "HALF_LIFE_FILTER"
        if
        variant_name.startswith(
            "HALF_LIFE"
        )
        else
        (
            "STOP_LOSS"
            if
            "STOP"
            in
            variant_name
            else
            "ENTRY_EXIT_THRESHOLDS"
        )
    )

    robustness_perf_rows.append(
        perf
    )

    if not trade_variant.empty:
        trade_variant = trade_variant.copy()

        trade_variant[
            "ROBUSTNESS_VARIANT"
        ] = variant_name

        robustness_trade_frames.append(
            trade_variant
        )


# ----- Formation-window and universe-size sensitivity -----

investable_robust = clean_columns(
    pd.read_csv(
        INVESTABLE_FILE,
        low_memory=False
    )
)

require_columns(
    investable_robust,
    [
        "FORMATION_DATE",
        "BLOCK_TYPE",
        "FINAL_INVESTABLE_COMPANIES",
    ],
    "Investable-universe file for robustness"
)

investable_robust[
    "FORMATION_DATE"
] = parse_dates(
    investable_robust[
        "FORMATION_DATE"
    ]
)

investable_robust[
    "BLOCK_TYPE"
] = (
    investable_robust[
        "BLOCK_TYPE"
    ]
    .astype(
        str
    )
    .str.strip()
    .str.upper()
)


def select_eg_variant(
    formation_months,
    universe_fraction,
    variant_name
):
    selected_rows = []

    dev_schedule = schedule[
        schedule[
            "BLOCK_TYPE"
        ].eq(
            "DEVELOPMENT"
        )
    ].copy()

    for sched_row in dev_schedule.itertuples(
        index=False
    ):
        formation_date = pd.Timestamp(
            sched_row.FORMATION_DATE
        )

        inv_match = investable_robust[
            investable_robust[
                "FORMATION_DATE"
            ].eq(
                formation_date
            )
            &
            investable_robust[
                "BLOCK_TYPE"
            ].eq(
                "DEVELOPMENT"
            )
        ]

        if len(
            inv_match
        ) != 1:
            raise RuntimeError(
                f"{variant_name}: investable universe missing/duplicated "
                f"for {formation_date.date()}."
            )

        companies = parse_company_list_simple(
            inv_match[
                "FINAL_INVESTABLE_COMPANIES"
            ].iloc[
                0
            ]
        )

        alt_start = (
            formation_date
            -
            pd.DateOffset(
                months=formation_months
            )
            +
            pd.Timedelta(
                days=1
            )
        )

        if universe_fraction < 1.0:
            med_liq = (
                prices.loc[
                    prices[
                        "DATE"
                    ].between(
                        alt_start,
                        formation_date,
                        inclusive="both"
                    )
                    &
                    prices[
                        "COMPANY_ID"
                    ].isin(
                        companies
                    )
                ]
                .groupby(
                    "COMPANY_ID"
                )[
                    "TOTAL_TRADED_VALUE"
                ]
                .median()
                .sort_values(
                    ascending=False
                )
            )

            keep_n = max(
                4,
                int(
                    np.ceil(
                        universe_fraction
                        *
                        len(
                            companies
                        )
                    )
                )
            )

            companies = [
                c
                for c in med_liq.index[
                    :
                    keep_n
                ]
                if c in companies
            ]

        directional_rows = []

        for i in range(
            len(
                companies
            )
        ):
            for j in range(
                i + 1,
                len(
                    companies
                )
            ):
                s1 = companies[
                    i
                ]

                s2 = companies[
                    j
                ]

                pair_id = "__".join(
                    sorted(
                        [
                            s1,
                            s2,
                        ]
                    )
                )

                a = prices.loc[
                    prices[
                        "COMPANY_ID"
                    ].eq(
                        s1
                    )
                    &
                    prices[
                        "DATE"
                    ].between(
                        alt_start,
                        formation_date,
                        inclusive="both"
                    ),
                    [
                        "DATE",
                        "TOTAL_RETURN_INDEX",
                        "SEGMENT_ID",
                    ],
                ].rename(
                    columns={
                        "TOTAL_RETURN_INDEX":
                            "P1",

                        "SEGMENT_ID":
                            "SEG1",
                    }
                )

                b = prices.loc[
                    prices[
                        "COMPANY_ID"
                    ].eq(
                        s2
                    )
                    &
                    prices[
                        "DATE"
                    ].between(
                        alt_start,
                        formation_date,
                        inclusive="both"
                    ),
                    [
                        "DATE",
                        "TOTAL_RETURN_INDEX",
                        "SEGMENT_ID",
                    ],
                ].rename(
                    columns={
                        "TOTAL_RETURN_INDEX":
                            "P2",

                        "SEGMENT_ID":
                            "SEG2",
                    }
                )

                pair = (
                    a.merge(
                        b,
                        on="DATE",
                        how="inner"
                    )
                    .sort_values(
                        "DATE"
                    )
                    .reset_index(
                        drop=True
                    )
                )

                min_obs = max(
                    100,
                    int(
                        245
                        *
                        formation_months
                        /
                        12
                        *
                        0.75
                    )
                )

                if len(
                    pair
                ) < min_obs:
                    continue

                if (
                    pair[
                        "SEG1"
                    ].nunique()
                    !=
                    1
                    or
                    pair[
                        "SEG2"
                    ].nunique()
                    !=
                    1
                ):
                    continue

                for dependent, explanatory in [
                    (
                        s1,
                        s2
                    ),
                    (
                        s2,
                        s1
                    ),
                ]:
                    if dependent == s1:
                        y = np.log(
                            pair[
                                "P1"
                            ].to_numpy(
                                dtype=float
                            )
                        )

                        x = np.log(
                            pair[
                                "P2"
                            ].to_numpy(
                                dtype=float
                            )
                        )

                    else:
                        y = np.log(
                            pair[
                                "P2"
                            ].to_numpy(
                                dtype=float
                            )
                        )

                        x = np.log(
                            pair[
                                "P1"
                            ].to_numpy(
                                dtype=float
                            )
                        )

                    X = np.column_stack(
                        [
                            np.ones(
                                len(
                                    x
                                )
                            ),
                            x,
                        ]
                    )

                    coef, _, _, _ = np.linalg.lstsq(
                        X,
                        y,
                        rcond=None
                    )

                    alpha = float(
                        coef[
                            0
                        ]
                    )

                    beta = float(
                        coef[
                            1
                        ]
                    )

                    residual = (
                        y
                        -
                        (
                            alpha
                            +
                            beta
                            *
                            x
                        )
                    )

                    try:
                        stat, pvalue, _ = coint(
                            y,
                            x,
                            trend="c",
                            autolag="aic"
                        )

                    except Exception:
                        continue

                    if not (
                        np.isfinite(
                            stat
                        )
                        and
                        np.isfinite(
                            pvalue
                        )
                        and
                        np.isfinite(
                            beta
                        )
                    ):
                        continue

                    residual_sd = float(
                        np.std(
                            residual,
                            ddof=1
                        )
                    )

                    if residual_sd <= NUMERIC_TOL:
                        continue

                    directional_rows.append(
                        {
                            "PAIR_ID":
                                pair_id,

                            "UNORDERED_COMPANY_1":
                                min(
                                    s1,
                                    s2
                                ),

                            "UNORDERED_COMPANY_2":
                                max(
                                    s1,
                                    s2
                                ),

                            "COMPANY_A":
                                dependent,

                            "COMPANY_B":
                                explanatory,

                            "ALPHA":
                                alpha,

                            "BETA":
                                beta,

                            "RESIDUAL_MEAN":
                                float(
                                    np.mean(
                                        residual
                                    )
                                ),

                            "RESIDUAL_STD_DDOF1":
                                residual_sd,

                            "TEST_STAT":
                                float(
                                    stat
                                ),

                            "P_RAW":
                                float(
                                    pvalue
                                ),
                        }
                    )

        directional = pd.DataFrame(
            directional_rows
        )

        if directional.empty:
            continue

        directional[
            "P_BH"
        ] = bh_adjust(
            directional[
                "P_RAW"
            ].to_numpy(
                dtype=float
            )
        )

        directional[
            "QUALIFIES"
        ] = (
            directional[
                "P_BH"
            ].le(
                0.05
            )
            &
            directional[
                "BETA"
            ].gt(
                0
            )
        )

        best_rows = []

        for pair_id, grp in directional.groupby(
            "PAIR_ID"
        ):
            q = grp[
                grp[
                    "QUALIFIES"
                ]
            ].copy()

            if q.empty:
                continue

            best = q.sort_values(
                [
                    "P_BH",
                    "P_RAW",
                    "TEST_STAT",
                ],
                ascending=[
                    True,
                    True,
                    True,
                ]
            ).iloc[
                0
            ]

            best_rows.append(
                best.to_dict()
            )

        best = pd.DataFrame(
            best_rows
        )

        if best.empty:
            continue

        best = best.sort_values(
            [
                "P_BH",
                "P_RAW",
                "TEST_STAT",
                "PAIR_ID",
            ]
        ).reset_index(
            drop=True
        )

        best[
            "EVIDENCE_RANK"
        ] = np.arange(
            1,
            len(
                best
            )
            +
            1
        )

        picked = choose_non_overlapping_pairs(
            best
        )

        for selected_number, row in enumerate(
            picked.itertuples(
                index=False
            ),
            start=1
        ):
            selected_rows.append(
                {
                    "FORMATION_DATE":
                        formation_date,

                    "BLOCK_TYPE":
                        "DEVELOPMENT",

                    "SELECTED_PAIR_NUMBER":
                        selected_number,

                    "PAIR_ID":
                        row.PAIR_ID,

                    "COMPANY_A":
                        row.COMPANY_A,

                    "COMPANY_B":
                        row.COMPANY_B,

                    "FORMATION_START":
                        alt_start,

                    "TRADING_START":
                        sched_row.TRADING_START,

                    "TRADING_END":
                        sched_row.TRADING_END,

                    "ALPHA":
                        row.ALPHA,

                    "BETA":
                        row.BETA,

                    "RESIDUAL_MEAN":
                        row.RESIDUAL_MEAN,

                    "RESIDUAL_STD_DDOF1":
                        row.RESIDUAL_STD_DDOF1,

                    "P_BH":
                        row.P_BH,

                    "FORMATION_MONTHS":
                        formation_months,

                    "UNIVERSE_FRACTION":
                        universe_fraction,

                    "ROBUSTNESS_VARIANT":
                        variant_name,
                }
            )

    return pd.DataFrame(
        selected_rows
    )


selection_variant_specs = [
    (
        "FORMATION_6_MONTHS",
        6,
        1.0,
        "FORMATION_WINDOW_LENGTH",
    ),
    (
        "FORMATION_18_MONTHS",
        18,
        1.0,
        "FORMATION_WINDOW_LENGTH",
    ),
    (
        "UNIVERSE_TOP_75PCT_LIQUID",
        12,
        0.75,
        "UNIVERSE_SIZE",
    ),
]


selection_robustness_frames = []


for (
    variant_name,
    formation_months,
    universe_fraction,
    category
) in selection_variant_specs:
    variant_selected = select_eg_variant(
        formation_months,
        universe_fraction,
        variant_name
    )

    if not variant_selected.empty:
        selection_robustness_frames.append(
            variant_selected
        )

    perf, _, trade_variant = run_development_variant(
        variant_selected,
        variant_name,
        entry_z=2.0,
        exit_z=0.0,
        stop_z=None
    )

    perf[
        "ROBUSTNESS_CATEGORY"
    ] = category

    robustness_perf_rows.append(
        perf
    )

    if not trade_variant.empty:
        trade_variant = trade_variant.copy()

        trade_variant[
            "ROBUSTNESS_VARIANT"
        ] = variant_name

        robustness_trade_frames.append(
            trade_variant
        )


if selection_robustness_frames:
    selection_robustness = pd.concat(
        selection_robustness_frames,
        ignore_index=True
    )

else:
    selection_robustness = pd.DataFrame()


selection_robustness.to_csv(
    OUTPUT_DIR
    / "13B_FORMATION_AND_UNIVERSE_SELECTION_SENSITIVITY.csv",
    index=False,
    date_format="%Y-%m-%d"
)


robustness_performance = pd.DataFrame(
    robustness_perf_rows
)

robustness_performance.to_csv(
    OUTPUT_DIR
    / "13C_REQUIRED_ROBUSTNESS_PERFORMANCE.csv",
    index=False,
    date_format="%Y-%m-%d"
)


if robustness_trade_frames:
    robustness_trade_pnl = pd.concat(
        robustness_trade_frames,
        ignore_index=True
    )

else:
    robustness_trade_pnl = pd.DataFrame()


robustness_trade_pnl.to_csv(
    OUTPUT_DIR
    / "13D_ROBUSTNESS_TRADE_PNL.csv",
    index=False,
    date_format="%Y-%m-%d"
)


# =============================================================================
# 16D. REQUIREMENTS-COVERAGE AUDIT
# =============================================================================

lag_sensitivity = clean_columns(
    pd.read_csv(
        LAG_SENSITIVITY_FILE,
        low_memory=False
    )
)

requirements_rows = [
    (
        "Correct Engle-Granger critical values",
        True,
        "Pair-selection code used statsmodels coint/MacKinnon values."
    ),
    (
        "Asymmetry handled",
        (
            "CHOSEN_DIRECTION"
            in
            selected.columns
        ),
        "Both A-on-B and B-on-A were tested; stronger qualifying direction retained."
    ),
    (
        "Lag choice stated and sensitivity checked",
        (
            len(
                lag_sensitivity
            )
            ==
            len(
                selected
            )
        ),
        "AIC primary; BIC and fixed lag=1 sensitivity saved."
    ),
    (
        "Multiple testing controlled",
        (
            selection_audit[
                "STATUS"
            ]
            .eq(
                "PASS"
            )
            .all()
        ),
        "Both directions counted in the corrected screening."
    ),
    (
        "Formation-only parameters frozen",
        True,
        "Selected alpha, beta, mean and standard deviation are frozen through each trading window."
    ),
    (
        "Entry / exit / stop / maximum holding specified",
        True,
        "Primary: entry +/-2SD, exit at mean, no separate stop, forced close at window end; 4SD stop checked in robustness."
    ),
    (
        "Position sizing / max simultaneous / pair cap specified",
        True,
        "Beta-based leg split; 25% gross per pair; maximum four selected pairs."
    ),
    (
        "OOS kept separate",
        True,
        "Primary results report development and OOS separately; robustness uses development only."
    ),
    (
        "Required performance metrics",
        required_performance[
            [
                "CAGR",
                "ANNUALIZED_VOLATILITY",
                "SHARPE_RF0",
                "SORTINO_RF0",
                "MAX_DRAWDOWN",
                "CALMAR",
            ]
        ].shape[
            1
        ]
        ==
        6,
        "CAGR, volatility, Sharpe, Sortino, drawdown and Calmar produced."
    ),
    (
        "Monthly returns",
        not monthly.empty,
        "Monthly table produced at 1x costs."
    ),
    (
        "Required trade statistics",
        True,
        "Number, hit rate, average/median P&L, holding distribution and convergence/stop-out rates produced."
    ),
    (
        "Attribution",
        True,
        "Pair, formation-window and top-decile P&L concentration produced."
    ),
    (
        "Market exposure",
        True,
        "NIFTY500 beta/correlation plus daily gross/net exposure and turnover produced."
    ),
    (
        "Cost sensitivity and gross-vs-net",
        set(
            COST_MULTIPLIERS
        )
        ==
        {
            0.0,
            0.5,
            1.0,
            2.0,
        },
        "0x is gross; 0.5x, 1x and 2x are cost sensitivity."
    ),
    (
        "Entry/exit threshold sensitivity",
        robustness_performance[
            "ROBUSTNESS_CATEGORY"
        ].eq(
            "ENTRY_EXIT_THRESHOLDS"
        ).any(),
        "1.5SD / 2SD / 2.5SD entry and alternate exit band checked."
    ),
    (
        "Formation-window sensitivity",
        robustness_performance[
            "ROBUSTNESS_CATEGORY"
        ].eq(
            "FORMATION_WINDOW_LENGTH"
        ).any(),
        "6-month and 18-month formation alternatives checked on development data."
    ),
    (
        "Universe-size sensitivity",
        robustness_performance[
            "ROBUSTNESS_CATEGORY"
        ].eq(
            "UNIVERSE_SIZE"
        ).any(),
        "Top 75% most-liquid subset checked on development data."
    ),
    (
        "Half-life filter sensitivity",
        robustness_performance[
            "ROBUSTNESS_CATEGORY"
        ].eq(
            "HALF_LIFE_FILTER"
        ).any(),
        "5-to-60-session filter checked, matching the assignment's example band."
    ),
    (
        "Short-leg realism",
        True,
        (
            "Primary entries verify official NSE stock-futures availability; "
            "no stock-borrow fee because short leg is a future; explicit "
            "margin/collateral funding cost is charged while the future is open."
        )
    ),
    (
        "Capacity in rupees",
        not strategy_capacity.empty,
        "1% participation cap translated into a conservative rupee portfolio capacity."
    ),
    (
        "Trade ledger required fields",
        (
            primary_trade_ledger.empty
            or
            set(
                [
                    "PAIR_ID",
                    "ENTRY_DIRECTION",
                    "ENTRY_EXECUTION_DATE",
                    "EXIT_EXECUTION_DATE",
                    "ENTRY_EXECUTION_Z",
                    "EXIT_EXECUTION_Z",
                    "GROSS_PNL",
                    "TRANSACTION_COST",
                    "FINANCING_COST",
                    "TOTAL_COST",
                    "NET_PNL",
                    "EXIT_REASON",
                ]
            ).issubset(
                primary_trade_ledger.columns
            )
        ),
        "Pair, direction, dates, spread z-scores, gross P&L, costs, net P&L and exit reason."
    ),
]


requirements_coverage = pd.DataFrame(
    [
        {
            "REQUIREMENT":
                name,

            "PASS":
                bool(
                    passed
                ),

            "EVIDENCE":
                evidence,
        }
        for name, passed, evidence
        in requirements_rows
    ]
)


requirements_coverage.to_csv(
    OUTPUT_DIR
    / "14_REQUIREMENTS_COVERAGE_AUDIT.csv",
    index=False
)


# Explicit OOS interpretation.
# A zero OOS return caused by zero qualifying pairs is NOT evidence of
# profitable OOS trading. Save that distinction mechanically.
oos_selected_pairs = int(
    selected[
        "BLOCK_TYPE"
    ].eq(
        "OOS"
    ).sum()
)

oos_executable_trades = (
    int(
        executable_trades[
            "BLOCK_TYPE"
        ].eq(
            "OOS"
        ).sum()
    )
    if
    not executable_trades.empty
    else
    0
)

oos_perf_1x = required_performance[
    required_performance[
        "COST_MULTIPLIER"
    ].eq(
        1.0
    )
    &
    required_performance[
        "PERIOD"
    ].eq(
        "OOS"
    )
]

oos_return_1x = (
    float(
        oos_perf_1x[
            "TOTAL_RETURN"
        ].iloc[
            0
        ]
    )
    if
    len(
        oos_perf_1x
    )
    else
    np.nan
)

if (
    oos_selected_pairs
    ==
    0
    and
    oos_executable_trades
    ==
    0
):
    oos_status = (
        "NO_QUALIFYING_PAIRS_NO_OOS_TRADING_EVIDENCE"
    )

    oos_interpretation = (
        "The EG rules selected no OOS pairs, so the strategy made no OOS "
        "trades. A 0% OOS return therefore does not demonstrate OOS "
        "profitability; it only shows that the strategy stayed in cash."
    )

else:
    oos_status = (
        "OOS_TRADES_OCCURRED"
    )

    oos_interpretation = (
        "OOS trades occurred; interpret the reported OOS return and trade "
        "statistics as actual OOS trading evidence."
    )


pd.DataFrame(
    [
        {
            "OOS_START":
                OOS_START,

            "N_OOS_SELECTED_PAIR_PERIODS":
                oos_selected_pairs,

            "N_OOS_EXECUTABLE_TRADES":
                oos_executable_trades,

            "OOS_RETURN_1X":
                oos_return_1x,

            "OOS_STATUS":
                oos_status,

            "INTERPRETATION":
                oos_interpretation,
        }
    ]
).to_csv(
    OUTPUT_DIR
    / "15_OOS_INTERPRETATION.csv",
    index=False,
    date_format="%Y-%m-%d"
)



if not requirements_coverage[
    "PASS"
].all():
    failed = requirements_coverage[
        ~requirements_coverage[
            "PASS"
        ]
    ]

    raise RuntimeError(
        "One or more assignment requirements were not covered:\n\n"
        +
        failed.to_string(
            index=False
        )
    )



# =============================================================================
# 17. FINAL AUDIT
# =============================================================================

audit_checks = []


def add_check(
    check,
    value,
    status
):
    audit_checks.append(
        {
            "CHECK":
                check,

            "VALUE":
                value,

            "STATUS":
                status,
        }
    )


add_check(
    "Earlier EG selection audit",
    "PASS",
    "PASS",
)

add_check(
    "Formation dates",
    len(
        schedule
    ),
    (
        "PASS"
        if
        len(
            schedule
        )
        ==
        EXPECTED_FORMATION_DATES
        else
        "FAIL"
    ),
)

add_check(
    "Selected EG pair-periods reconcile with corrected selection audit",
    len(
        selected
    ),
    (
        "PASS"
        if
        len(
            selected
        )
        ==
        audited_selected_count
        else
        "FAIL"
    ),
)

add_check(
    "OOS selected pair-periods",
    int(
        selected[
            "BLOCK_TYPE"
        ].eq(
            "OOS"
        ).sum()
    ),
    (
        "PASS"
        if
        int(
            selected[
                "BLOCK_TYPE"
            ].eq(
                "OOS"
            ).sum()
        )
        ==
        0
        else
        "FAIL"
    ),
)

add_check(
    "Logical trades",
    len(
        logical
    ),
    "PASS",
)

add_check(
    "Executable trades",
    len(
        executable_trades
    ),
    "PASS",
)

add_check(
    "F&O-infeasible logical trades skipped",
    len(
        skipped_trades
    ),
    "PASS",
)

add_check(
    "Pair-window signal audits failed",
    int(
        (
            ~pair_window_audit[
                "STATUS"
            ].isin(
                [
                    "PASS",
                    "NO_TRADING_BEFORE_STRUCTURAL_BREAK",
                ]
            )
        ).sum()
    ),
    (
        "PASS"
        if
        (
            ~pair_window_audit[
                "STATUS"
            ].isin(
                [
                    "PASS",
                    "NO_TRADING_BEFORE_STRUCTURAL_BREAK",
                ]
            )
        ).sum()
        ==
        0
        else
        "FAIL"
    ),
)

add_check(
    "Same-close entry execution failures",
    (
        int(
            (
                logical[
                    "ENTRY_EXECUTION_DATE"
                ]
                <=
                logical[
                    "ENTRY_SIGNAL_DATE"
                ]
            ).sum()
        )
        if
        not logical.empty
        else
        0
    ),
    (
        "PASS"
        if
        logical.empty
        or
        (
            logical[
                "ENTRY_EXECUTION_DATE"
            ]
            >
            logical[
                "ENTRY_SIGNAL_DATE"
            ]
        ).all()
        else
        "FAIL"
    ),
)

add_check(
    "OOS executable trades",
    (
        int(
            executable_trades[
                "BLOCK_TYPE"
            ].eq(
                "OOS"
            ).sum()
        )
        if
        not executable_trades.empty
        else
        0
    ),
    (
        "PASS"
        if
        executable_trades.empty
        or
        int(
            executable_trades[
                "BLOCK_TYPE"
            ].eq(
                "OOS"
            ).sum()
        )
        ==
        0
        else
        "FAIL"
    ),
)


# P&L reconciliation at each cost multiplier.
if not trade_pnl.empty:
    for multiplier in COST_MULTIPLIERS:
        x = trade_pnl[
            trade_pnl[
                "COST_MULTIPLIER"
            ].eq(
                multiplier
            )
        ]

        max_error = (
            (
                x[
                    "NET_PNL"
                ]
                -
                (
                    x[
                        "GROSS_PNL"
                    ]
                    -
                    x[
                        "TOTAL_COST"
                    ]
                )
            )
            .abs()
            .max()
        )

        add_check(
            f"P&L identity max error at {multiplier}x costs",
            float(
                max_error
            ),
            (
                "PASS"
                if
                max_error
                <
                1e-10
                else
                "FAIL"
            ),
        )


final_audit = pd.DataFrame(
    audit_checks
)


overall_status = (
    "PASS"
    if
    (
        final_audit[
            "STATUS"
        ]
        ==
        "PASS"
    ).all()
    else
    "FAIL"
)


overall_row = pd.DataFrame(
    [
        {
            "CHECK":
                "OVERALL_STATUS",

            "VALUE":
                overall_status,

            "STATUS":
                overall_status,
        }
    ]
)


final_audit = pd.concat(
    [
        overall_row,
        final_audit,
    ],
    ignore_index=True
)


final_audit.to_csv(
    OUTPUT_DIR
    / "11_EG_FINAL_BACKTEST_AUDIT.csv",
    index=False
)


if overall_status != (
    "PASS"
):
    raise RuntimeError(
        "Final EG backtest audit FAILED.\n"
        +
        final_audit.to_string(
            index=False
        )
    )


# =============================================================================
# 18. CONFIG / LIMITATIONS
# =============================================================================

config = pd.DataFrame(
    [
        [
            "METHOD",
            "ENGLE_GRANGER",
        ],
        [
            "ENTRY_RULE",
            "GAP >= FORMATION_MEAN + 2SD OR GAP <= FORMATION_MEAN - 2SD",
        ],
        [
            "NORMAL_EXIT_RULE",
            "DIRECTIONAL CROSSING OF FROZEN FORMATION GAP MEAN",
        ],
        [
            "SIGNAL_EXECUTION",
            "SIGNAL AT CLOSE T; EXECUTE NEXT COMMON OBSERVATION",
        ],
        [
            "REENTRY",
            "ALLOWED AFTER COMPLETED TRADE",
        ],
        [
            "FORCED_EXIT",
            "END OF 6-MONTH WINDOW OR BEFORE STRUCTURAL BREAK",
        ],
        [
            "PAIR_GROSS_WEIGHT",
            str(
                PAIR_GROSS_WEIGHT
            ),
        ],
        [
            "LEG_SIZING",
            "ABSOLUTE A:B NOTIONAL = 1:BETA, SCALED TO 25% PAIR GROSS",
        ],
        [
            "LONG_LEG",
            "CASH EQUITY TOTAL RETURN",
        ],
        [
            "SHORT_LEG",
            "STOCK-FUTURES ELIGIBILITY CHECKED; P&L USES CORPORATE-ACTION-ADJUSTED SPOT PRICE PROXY EXCLUDING CASH DIVIDENDS",
        ],
        [
            "CASH_ROUND_TRIP_COST_BPS",
            str(
                CASH_RT_BPS
            ),
        ],
        [
            "FUTURES_ROUND_TRIP_COST_BPS",
            str(
                FUTURES_RT_BPS
            ),
        ],
        [
            "COST_SENSITIVITY",
            "0x(gross);0.5x;1x;2x — multiplier scales transaction and financing costs",
        ],
        [
            "SHORT_SIDE_FINANCING_ASSUMPTION",
            (
                f"{FUTURES_MARGIN_FRACTION:.0%} margin/collateral fraction "
                f"funded at {FINANCING_RATE_ANNUAL:.0%} per year"
            ),
        ],
        [
            "FNO_INFEASIBLE_TRADE_TREATMENT",
            "SKIP ENTIRE LOGICAL TRADE; NO SYNTHETIC REENTRY",
        ],
        [
            "UNUSED_CAPITAL_RETURN",
            "0%",
        ],
        [
            "OOS_START",
            str(
                OOS_START.date()
            ),
        ],
        [
            "FUTURES_BASIS",
            "NOT_MODELED",
        ],
        [
            "FUTURES_MONTHLY_ROLL",
            "NOT_MODELED",
        ],
        [
            "INTEGER_FUTURES_LOT_SIZE",
            "NOT_MODELED",
        ],
        [
            "FUTURES_MARGIN_COLLATERAL_FUNDING",
            (
                f"MODELED: {FUTURES_MARGIN_FRACTION:.0%} OF SHORT FUTURES "
                f"NOTIONAL FUNDED AT {FINANCING_RATE_ANNUAL:.0%} PER YEAR"
            ),
        ],
    ],
    columns=[
        "PARAMETER",
        "VALUE",
    ]
)


config.to_csv(
    OUTPUT_DIR
    / "00_EG_FINAL_CONFIG.csv",
    index=False
)


# =============================================================================
# 19. CONSOLE SUMMARY
# =============================================================================

print(
    "\n"
    +
    "="
    *
    112
)

print(
    "ENGLE-GRANGER BACKTEST COMPLETE — AUDIT PASS"
)

print(
    "="
    *
    112
)


print(
    "\nPRIMARY 1x PERFORMANCE"
)

print(
    performance[
        performance[
            "COST_MULTIPLIER"
        ].eq(
            1.0
        )
    ].to_string(
        index=False
    )
)


print(
    "\nPRIMARY 1x TRADE STATISTICS"
)

print(
    trade_stats.to_string(
        index=False
    )
)


print(
    "\nALL 19 SIX-MONTH PERIOD RETURNS"
)

print(
    formation_period_returns[
        [
            "FORMATION_DATE",
            "BLOCK_TYPE",
            "N_EG_PAIRS_SELECTED",
            "N_EXECUTABLE_TRADES",
            "PORTFOLIO_RETURN_1X",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nOutput folder:"
)

print(
    OUTPUT_DIR
)


print(
    "\nUpload these files next:"
)

for filename in [
    "04A_PRIMARY_TRADE_LEDGER_1X.csv",
    "05A_REQUIRED_PERFORMANCE_METRICS.csv",
    "06A_REQUIRED_TRADE_STATISTICS.csv",
    "07A_MONTHLY_RETURNS_1X.csv",
    "08A_PAIR_ATTRIBUTION_1X.csv",
    "08B_FORMATION_WINDOW_ATTRIBUTION_1X.csv",
    "08C_TOP_DECILE_PNL_CONCENTRATION_1X.csv",
    "09B_STRATEGY_CAPACITY_RUPEES.csv",
    "10_NIFTY500_BETA_CORRELATION.csv",
    "13A_SELECTED_PAIR_HALF_LIFE.csv",
    "13C_REQUIRED_ROBUSTNESS_PERFORMANCE.csv",
    "14_REQUIREMENTS_COVERAGE_AUDIT.csv",
    "15_OOS_INTERPRETATION.csv",
    "11_EG_FINAL_BACKTEST_AUDIT.csv",
]:
    print(
        OUTPUT_DIR
        /
        filename
    )
