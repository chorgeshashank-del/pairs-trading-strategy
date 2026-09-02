import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import coint
except ImportError as exc:
    raise ImportError(
        "This program needs statsmodels. Install it once with:\n"
        "pip install statsmodels"
    ) from exc


# =============================================================================
# NIFTY PHARMA — EG PAIR SELECTION, BOTH DIRECTIONS
# =============================================================================
#
# PURPOSE
# -------
# Fix the Engle-Granger asymmetry issue properly:
#
# For every unordered stock pair {X, Y}, test BOTH:
#
#     log(X) = alpha + beta*log(Y) + gap
#
# and
#
#     log(Y) = alpha + beta*log(X) + gap
#
# The main result uses AIC to choose the number of lagged gap changes in the
# Engle-Granger test. BIC and fixed lag=1 are also calculated as sensitivity
# checks because the assignment explicitly requires this.
#
# IMPORTANT:
# We count BOTH directions when correcting for the fact that many tests are
# performed. This prevents "try both and keep the nicer one" from making the
# test artificially easy.
#
# Main selection rule:
#   1. Run BOTH directions for every pair using formation data only.
#   2. Apply Benjamini-Hochberg 5% correction to ALL directional AIC tests
#      on that formation date.
#   3. For each unordered pair, among directions that pass the corrected
#      5% rule and have beta > 0, keep the stronger direction.
#   4. Rank surviving unordered pairs by corrected AIC p-value, then raw
#      AIC p-value, then test statistic.
#   5. Select up to 4 non-overlapping pairs.
#
# Output remains compatible with the later EG backtest:
#   COMPANY_A = dependent stock in the CHOSEN direction
#   COMPANY_B = explanatory stock in the CHOSEN direction
#
# This program DOES NOT generate trades or calculate P&L.
# =============================================================================


# =============================================================================
# 1. PATHS / SETTINGS
# =============================================================================

PROJECT_ROOT = Path(
    os.environ.get(
        "PAIR_TRADING_PROJECT_ROOT",
        r"C:\fin proj"
    )
)

INVESTABLE_FILE = (
    PROJECT_ROOT
    / "nse_pharma_final_investable_universe_FINAL"
    / "07_FINAL_INVESTABLE_BY_FORMATION.csv"
)

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

EG_ROOT = (
    PROJECT_ROOT
    / "pair_trading_methods"
    / "ENGLE_GRANGER"
)

PAIR_SELECTION_DIR = (
    EG_ROOT
    / "01_pair_selection"
)

AUDIT_DIR = (
    EG_ROOT
    / "04_audit"
)

PAIR_SELECTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FDR_LEVEL = 0.05
MAX_SELECTED_PAIRS = 4
MIN_COMMON_OBSERVATIONS = 100
REQUIRE_POSITIVE_BETA = True
FIXED_LAG = 1
NUMERIC_TOL = 1e-12


# =============================================================================
# 2. HELPERS
# =============================================================================

def clean_columns(df):
    out = df.copy()

    out.columns = (
        out.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.upper()
    )

    return out


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
        set(required)
        -
        set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{name} missing required columns: {sorted(missing)}"
        )


def parse_company_list(value):
    if pd.isna(value):
        return []

    values = [
        x.strip().upper()
        for x in str(value).split(";")
        if x.strip()
    ]

    return sorted(
        set(values)
    )


def unordered_pair_id(
    x,
    y
):
    a, b = sorted(
        [
            str(x).upper(),
            str(y).upper(),
        ]
    )

    return (
        f"{a}__{b}"
    )


def benjamini_hochberg(
    pvalues
):
    """
    Adjust p-values for the fact that many tests were run.

    Plain English:
    when we test many relationships, some can look good by luck.
    This correction makes qualification harder when more tests are run.
    """

    p = np.asarray(
        pvalues,
        dtype=float
    )

    q = np.full(
        len(p),
        np.nan,
        dtype=float
    )

    valid = np.isfinite(
        p
    )

    if not valid.any():
        return q

    pv = p[
        valid
    ]

    m = len(
        pv
    )

    order = np.argsort(
        pv
    )

    ranked = pv[
        order
    ]

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
        adjusted[::-1]
    )[::-1]

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


def estimate_relationship(
    log_dependent,
    log_explanatory
):
    """
    Estimate:
        log(dependent) = alpha + beta*log(explanatory) + residual
    """

    y = np.asarray(
        log_dependent,
        dtype=float
    )

    x = np.asarray(
        log_explanatory,
        dtype=float
    )

    X = np.column_stack(
        [
            np.ones(
                len(x)
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
        coef[0]
    )

    beta = float(
        coef[1]
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

    return (
        alpha,
        beta,
        residual,
    )


def run_coint_test(
    y,
    x,
    mode
):
    """
    mode:
      AIC     -> automatic lag choice using AIC
      BIC     -> automatic lag choice using BIC
      FIXED1  -> fixed lag = 1
    """

    if mode == "AIC":
        stat, pvalue, crit = coint(
            y,
            x,
            trend="c",
            autolag="aic"
        )

    elif mode == "BIC":
        stat, pvalue, crit = coint(
            y,
            x,
            trend="c",
            autolag="bic"
        )

    elif mode == "FIXED1":
        stat, pvalue, crit = coint(
            y,
            x,
            trend="c",
            maxlag=FIXED_LAG,
            autolag=None
        )

    else:
        raise ValueError(
            f"Unknown test mode: {mode}"
        )

    return (
        float(stat),
        float(pvalue),
        np.asarray(
            crit,
            dtype=float
        ),
    )


def exact_disjoint_selection(
    eligible_df,
    max_pairs
):
    """
    Select as many non-overlapping pairs as possible, up to max_pairs.
    Among choices with the same number of pairs, prefer the set with the
    smallest total evidence rank.
    """

    if eligible_df.empty:
        return (
            [],
            0,
        )

    companies = sorted(
        set(
            eligible_df[
                "UNORDERED_COMPANY_1"
            ]
        )
        |
        set(
            eligible_df[
                "UNORDERED_COMPANY_2"
            ]
        )
    )

    company_to_index = {
        company:
            i
        for i, company
        in enumerate(
            companies
        )
    }

    edges = {}

    for row in eligible_df.itertuples(
        index=False
    ):
        u = company_to_index[
            row.UNORDERED_COMPANY_1
        ]

        v = company_to_index[
            row.UNORDERED_COMPANY_2
        ]

        if u > v:
            u, v = v, u

        edges[
            (
                u,
                v,
            )
        ] = (
            float(
                row.EVIDENCE_RANK
            ),
            row.PAIR_ID,
        )

    neighbors = {
        i: []
        for i in range(
            len(companies)
        )
    }

    for (
        u,
        v
    ), (
        weight,
        pair_id
    ) in edges.items():
        neighbors[
            u
        ].append(
            (
                v,
                weight,
                pair_id,
            )
        )

        neighbors[
            v
        ].append(
            (
                u,
                weight,
                pair_id,
            )
        )

    full_mask = (
        1
        <<
        len(companies)
    ) - 1

    INF = float(
        "inf"
    )

    @lru_cache(
        maxsize=None
    )
    def solve(
        mask,
        k
    ):
        if k == 0:
            return (
                0.0,
                tuple(),
            )

        if mask.bit_count() < 2 * k:
            return (
                INF,
                tuple(),
            )

        if mask == 0:
            return (
                INF,
                tuple(),
            )

        u = (
            mask
            &
            -mask
        ).bit_length() - 1

        # Leave u unused.
        best_cost, best_pairs = solve(
            mask
            &
            ~(
                1
                <<
                u
            ),
            k
        )

        # Pair u with one available neighbor.
        for (
            v,
            weight,
            pair_id
        ) in neighbors[
            u
        ]:
            if not (
                mask
                &
                (
                    1
                    <<
                    v
                )
            ):
                continue

            next_mask = (
                mask
                &
                ~(
                    1
                    <<
                    u
                )
                &
                ~(
                    1
                    <<
                    v
                )
            )

            sub_cost, sub_pairs = solve(
                next_mask,
                k - 1
            )

            if not np.isfinite(
                sub_cost
            ):
                continue

            candidate_cost = (
                weight
                +
                sub_cost
            )

            candidate_pairs = tuple(
                sorted(
                    (
                        pair_id,
                    )
                    +
                    sub_pairs
                )
            )

            if (
                candidate_cost
                <
                best_cost
                -
                NUMERIC_TOL
            ):
                best_cost = (
                    candidate_cost
                )

                best_pairs = (
                    candidate_pairs
                )

            elif (
                abs(
                    candidate_cost
                    -
                    best_cost
                )
                <=
                NUMERIC_TOL
                and
                candidate_pairs
                <
                best_pairs
            ):
                best_pairs = (
                    candidate_pairs
                )

        return (
            best_cost,
            best_pairs,
        )

    max_possible = min(
        max_pairs,
        len(companies) // 2
    )

    for k in range(
        max_possible,
        -1,
        -1
    ):
        cost, pair_ids = solve(
            full_mask,
            k
        )

        if np.isfinite(
            cost
        ):
            return (
                list(
                    pair_ids
                ),
                k,
            )

    return (
        [],
        0,
    )


# =============================================================================
# 3. LOAD INPUTS
# =============================================================================

print(
    "="
    *
    116
)

print(
    "ENGLE-GRANGER — BOTH-DIRECTION PAIR SELECTION"
)

print(
    "="
    *
    116
)


for path in [
    INVESTABLE_FILE,
    SCHEDULE_FILE,
    TOTAL_RETURN_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found:\n{path}"
        )


investable = clean_columns(
    pd.read_csv(
        INVESTABLE_FILE,
        low_memory=False
    )
)

schedule_raw = clean_columns(
    pd.read_csv(
        SCHEDULE_FILE,
        low_memory=False
    )
)

tr = clean_columns(
    pd.read_csv(
        TOTAL_RETURN_FILE,
        low_memory=False
    )
)


require_columns(
    investable,
    [
        "FORMATION_DATE",
        "BLOCK_TYPE",
        "N_FINAL_INVESTABLE",
        "FINAL_INVESTABLE_COMPANIES",
    ],
    "Final investable universe",
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
    "Frozen schedule",
)

require_columns(
    tr,
    [
        "DATE",
        "COMPANY_ID",
        "TOTAL_RETURN_INDEX",
        "SEGMENT_ID",
    ],
    "Total-return data",
)


investable[
    "FORMATION_DATE"
] = parse_dates(
    investable[
        "FORMATION_DATE"
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

tr[
    "DATE"
] = parse_dates(
    tr[
        "DATE"
    ]
)


for df, cols in [
    (
        investable,
        [
            "BLOCK_TYPE",
        ],
    ),
    (
        schedule_raw,
        [
            "BLOCK_TYPE",
        ],
    ),
    (
        tr,
        [
            "COMPANY_ID",
            "SEGMENT_ID",
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
            .astype(str)
            .str.strip()
            .str.upper()
        )


tr[
    "TOTAL_RETURN_INDEX"
] = pd.to_numeric(
    tr[
        "TOTAL_RETURN_INDEX"
    ],
    errors="coerce"
)


if investable[
    "FORMATION_DATE"
].isna().any():
    raise RuntimeError(
        "Invalid formation date in investable universe."
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
        "Invalid date in schedule."
    )

if tr[
    [
        "DATE",
        "COMPANY_ID",
        "TOTAL_RETURN_INDEX",
    ]
].isna().any().any():
    raise RuntimeError(
        "Missing/invalid core value in total-return data."
    )

if (
    tr[
        "TOTAL_RETURN_INDEX"
    ]
    <=
    0
).any():
    raise RuntimeError(
        "TOTAL_RETURN_INDEX must be positive."
    )

if tr[
    [
        "DATE",
        "COMPANY_ID",
    ]
].duplicated().any():
    raise RuntimeError(
        "Duplicate DATE + COMPANY_ID in total-return data."
    )

if investable[
    "FORMATION_DATE"
].duplicated().any():
    raise RuntimeError(
        "Duplicate formation date in investable universe."
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
        "Schedule has inconsistent duplicate formation dates."
    )


formation_table = schedule.merge(
    investable[
        [
            "FORMATION_DATE",
            "BLOCK_TYPE",
            "N_FINAL_INVESTABLE",
            "FINAL_INVESTABLE_COMPANIES",
        ]
    ],
    on=[
        "FORMATION_DATE",
        "BLOCK_TYPE",
    ],
    how="inner",
    validate="one_to_one"
)

if len(
    formation_table
) != len(
    schedule
):
    raise RuntimeError(
        "Schedule and investable universe do not match."
    )


# =============================================================================
# 4. RUN BOTH DIRECTIONS FOR EVERY PAIR
# =============================================================================

directional_rows = []
input_audit_rows = []
residual_cache = {}


for formation_number, formation_row in enumerate(
    formation_table.itertuples(
        index=False
    ),
    start=1
):
    formation_date = pd.Timestamp(
        formation_row.FORMATION_DATE
    )

    formation_start = pd.Timestamp(
        formation_row.FORMATION_START
    )

    companies = parse_company_list(
        formation_row.FINAL_INVESTABLE_COMPANIES
    )

    if len(
        companies
    ) != int(
        formation_row.N_FINAL_INVESTABLE
    ):
        raise RuntimeError(
            f"{formation_date.date()}: saved investable count does not "
            "match saved company list."
        )


    window = tr[
        tr[
            "DATE"
        ].between(
            formation_start,
            formation_date,
            inclusive="both"
        )
        &
        tr[
            "COMPANY_ID"
        ].isin(
            companies
        )
    ].copy()


    if window.empty:
        raise RuntimeError(
            f"{formation_date.date()}: empty formation window."
        )


    if window[
        "DATE"
    ].max() > formation_date:
        raise RuntimeError(
            f"{formation_date.date()}: future data entered formation window."
        )


    n_unordered_pairs = (
        len(companies)
        *
        (
            len(companies)
            -
            1
        )
        //
        2
    )

    expected_directional_tests = (
        2
        *
        n_unordered_pairs
    )


    print(
        f"[{formation_number:02d}/{len(formation_table):02d}] "
        f"{formation_date.date()} | stocks={len(companies)} | "
        f"unordered pairs={n_unordered_pairs} | "
        f"directional tests={expected_directional_tests}",
        flush=True
    )


    date_rows = []


    for i in range(
        len(companies)
    ):
        for j in range(
            i + 1,
            len(companies)
        ):
            stock_1 = companies[
                i
            ]

            stock_2 = companies[
                j
            ]

            pair_id = unordered_pair_id(
                stock_1,
                stock_2
            )


            s1 = window[
                window[
                    "COMPANY_ID"
                ].eq(
                    stock_1
                )
            ][
                [
                    "DATE",
                    "TOTAL_RETURN_INDEX",
                    "SEGMENT_ID",
                ]
            ].rename(
                columns={
                    "TOTAL_RETURN_INDEX":
                        "TRI_1",

                    "SEGMENT_ID":
                        "SEGMENT_1",
                }
            )

            s2 = window[
                window[
                    "COMPANY_ID"
                ].eq(
                    stock_2
                )
            ][
                [
                    "DATE",
                    "TOTAL_RETURN_INDEX",
                    "SEGMENT_ID",
                ]
            ].rename(
                columns={
                    "TOTAL_RETURN_INDEX":
                        "TRI_2",

                    "SEGMENT_ID":
                        "SEGMENT_2",
                }
            )


            pair = (
                s1.merge(
                    s2,
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


            n_obs = len(
                pair
            )

            segments_1 = sorted(
                pair[
                    "SEGMENT_1"
                ].dropna().astype(
                    str
                ).unique()
            )

            segments_2 = sorted(
                pair[
                    "SEGMENT_2"
                ].dropna().astype(
                    str
                ).unique()
            )


            for dependent, explanatory in [
                (
                    stock_1,
                    stock_2,
                ),
                (
                    stock_2,
                    stock_1,
                ),
            ]:
                direction_label = (
                    f"{dependent}_ON_{explanatory}"
                )

                row = {
                    "FORMATION_DATE":
                        formation_date,

                    "BLOCK_TYPE":
                        formation_row.BLOCK_TYPE,

                    "FORMATION_START":
                        formation_start,

                    "TRADING_START":
                        formation_row.TRADING_START,

                    "TRADING_END":
                        formation_row.TRADING_END,

                    "PAIR_ID":
                        pair_id,

                    "UNORDERED_COMPANY_1":
                        min(
                            stock_1,
                            stock_2
                        ),

                    "UNORDERED_COMPANY_2":
                        max(
                            stock_1,
                            stock_2
                        ),

                    "DIRECTION":
                        direction_label,

                    "DEPENDENT_COMPANY":
                        dependent,

                    "EXPLANATORY_COMPANY":
                        explanatory,

                    "COMMON_OBSERVATIONS":
                        n_obs,

                    "FIRST_COMMON_DATE":
                        (
                            pair[
                                "DATE"
                            ].min()
                            if
                            n_obs
                            else
                            pd.NaT
                        ),

                    "LAST_COMMON_DATE":
                        (
                            pair[
                                "DATE"
                            ].max()
                            if
                            n_obs
                            else
                            pd.NaT
                        ),

                    "VALID_TEST":
                        False,

                    "INVALID_REASON":
                        "",

                    "ALPHA":
                        np.nan,

                    "BETA":
                        np.nan,

                    "RESIDUAL_MEAN":
                        np.nan,

                    "RESIDUAL_STD_DDOF1":
                        np.nan,

                    "AIC_TEST_STAT":
                        np.nan,

                    "AIC_PVALUE_RAW":
                        np.nan,

                    "AIC_CRITICAL_1PCT":
                        np.nan,

                    "AIC_CRITICAL_5PCT":
                        np.nan,

                    "AIC_CRITICAL_10PCT":
                        np.nan,

                    "BIC_TEST_STAT":
                        np.nan,

                    "BIC_PVALUE_RAW":
                        np.nan,

                    "FIXED1_TEST_STAT":
                        np.nan,

                    "FIXED1_PVALUE_RAW":
                        np.nan,

                    "SEGMENTS_STOCK_1":
                        ";".join(
                            segments_1
                        ),

                    "SEGMENTS_STOCK_2":
                        ";".join(
                            segments_2
                        ),
                }


                if n_obs < MIN_COMMON_OBSERVATIONS:
                    row[
                        "INVALID_REASON"
                    ] = (
                        "TOO_FEW_COMMON_OBSERVATIONS"
                    )

                    date_rows.append(
                        row
                    )

                    continue


                if (
                    len(
                        segments_1
                    )
                    !=
                    1
                    or
                    len(
                        segments_2
                    )
                    !=
                    1
                ):
                    row[
                        "INVALID_REASON"
                    ] = (
                        "STRUCTURAL_SEGMENT_CHANGE_IN_FORMATION"
                    )

                    date_rows.append(
                        row
                    )

                    continue


                if dependent == stock_1:
                    log_y = np.log(
                        pair[
                            "TRI_1"
                        ].to_numpy(
                            dtype=float
                        )
                    )

                    log_x = np.log(
                        pair[
                            "TRI_2"
                        ].to_numpy(
                            dtype=float
                        )
                    )

                else:
                    log_y = np.log(
                        pair[
                            "TRI_2"
                        ].to_numpy(
                            dtype=float
                        )
                    )

                    log_x = np.log(
                        pair[
                            "TRI_1"
                        ].to_numpy(
                            dtype=float
                        )
                    )


                if (
                    np.std(
                        log_y,
                        ddof=1
                    )
                    <=
                    NUMERIC_TOL
                    or
                    np.std(
                        log_x,
                        ddof=1
                    )
                    <=
                    NUMERIC_TOL
                ):
                    row[
                        "INVALID_REASON"
                    ] = (
                        "NEAR_CONSTANT_LOG_SERIES"
                    )

                    date_rows.append(
                        row
                    )

                    continue


                try:
                    alpha, beta, residual = (
                        estimate_relationship(
                            log_y,
                            log_x
                        )
                    )

                    aic_stat, aic_p, aic_crit = (
                        run_coint_test(
                            log_y,
                            log_x,
                            "AIC"
                        )
                    )

                    bic_stat, bic_p, _ = (
                        run_coint_test(
                            log_y,
                            log_x,
                            "BIC"
                        )
                    )

                    fixed_stat, fixed_p, _ = (
                        run_coint_test(
                            log_y,
                            log_x,
                            "FIXED1"
                        )
                    )

                except Exception as exc:
                    row[
                        "INVALID_REASON"
                    ] = (
                        "TEST_ERROR_"
                        +
                        type(
                            exc
                        ).__name__
                    )

                    date_rows.append(
                        row
                    )

                    continue


                residual_std = float(
                    np.std(
                        residual,
                        ddof=1
                    )
                )


                required_numbers = [
                    alpha,
                    beta,
                    residual_std,
                    aic_p,
                    bic_p,
                    fixed_p,
                ]


                if not all(
                    np.isfinite(
                        v
                    )
                    for v in required_numbers
                ):
                    row[
                        "INVALID_REASON"
                    ] = (
                        "NONFINITE_RESULT"
                    )

                    date_rows.append(
                        row
                    )

                    continue


                if residual_std <= NUMERIC_TOL:
                    row[
                        "INVALID_REASON"
                    ] = (
                        "ZERO_RESIDUAL_STD"
                    )

                    date_rows.append(
                        row
                    )

                    continue


                row.update(
                    {
                        "VALID_TEST":
                            True,

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
                            residual_std,

                        "AIC_TEST_STAT":
                            aic_stat,

                        "AIC_PVALUE_RAW":
                            aic_p,

                        "AIC_CRITICAL_1PCT":
                            float(
                                aic_crit[
                                    0
                                ]
                            ),

                        "AIC_CRITICAL_5PCT":
                            float(
                                aic_crit[
                                    1
                                ]
                            ),

                        "AIC_CRITICAL_10PCT":
                            float(
                                aic_crit[
                                    2
                                ]
                            ),

                        "BIC_TEST_STAT":
                            bic_stat,

                        "BIC_PVALUE_RAW":
                            bic_p,

                        "FIXED1_TEST_STAT":
                            fixed_stat,

                        "FIXED1_PVALUE_RAW":
                            fixed_p,
                    }
                )


                date_rows.append(
                    row
                )


                residual_cache[
                    (
                        formation_date,
                        pair_id,
                        direction_label,
                    )
                ] = pd.DataFrame(
                    {
                        "FORMATION_DATE":
                            formation_date,

                        "BLOCK_TYPE":
                            formation_row.BLOCK_TYPE,

                        "PAIR_ID":
                            pair_id,

                        "DIRECTION":
                            direction_label,

                        "COMPANY_A":
                            dependent,

                        "COMPANY_B":
                            explanatory,

                        "DATE":
                            pair[
                                "DATE"
                            ].to_numpy(),

                        "LOG_A":
                            log_y,

                        "LOG_B":
                            log_x,

                        "RESIDUAL_SPREAD":
                            residual,
                    }
                )


    date_df = pd.DataFrame(
        date_rows
    )


    if len(
        date_df
    ) != expected_directional_tests:
        raise RuntimeError(
            f"{formation_date.date()}: expected "
            f"{expected_directional_tests} directional rows; "
            f"found {len(date_df)}."
        )


    valid = date_df[
        "VALID_TEST"
    ].eq(
        True
    )


    # -------------------------------------------------------------------------
    # Correct ALL directional tests, not one direction per pair.
    # We do this independently for the AIC main test and the two sensitivity
    # versions so the sensitivity comparison is like-for-like.
    # -------------------------------------------------------------------------

    for raw_col, adjusted_col in [
        (
            "AIC_PVALUE_RAW",
            "AIC_PVALUE_BH",
        ),
        (
            "BIC_PVALUE_RAW",
            "BIC_PVALUE_BH",
        ),
        (
            "FIXED1_PVALUE_RAW",
            "FIXED1_PVALUE_BH",
        ),
    ]:
        date_df[
            adjusted_col
        ] = np.nan

        if valid.any():
            date_df.loc[
                valid,
                adjusted_col,
            ] = benjamini_hochberg(
                date_df.loc[
                    valid,
                    raw_col,
                ].to_numpy(
                    dtype=float
                )
            )


    date_df[
        "PASSES_AIC_BH_5PCT"
    ] = (
        valid
        &
        date_df[
            "AIC_PVALUE_BH"
        ].le(
            FDR_LEVEL
        )
    )


    date_df[
        "PASSES_BIC_BH_5PCT"
    ] = (
        valid
        &
        date_df[
            "BIC_PVALUE_BH"
        ].le(
            FDR_LEVEL
        )
    )


    date_df[
        "PASSES_FIXED1_BH_5PCT"
    ] = (
        valid
        &
        date_df[
            "FIXED1_PVALUE_BH"
        ].le(
            FDR_LEVEL
        )
    )


    date_df[
        "POSITIVE_BETA"
    ] = (
        date_df[
            "BETA"
        ]
        >
        0
    )


    # Primary eligibility = AIC main test + corrected 5% + positive beta.
    date_df[
        "DIRECTION_ELIGIBLE_PRIMARY"
    ] = (
        date_df[
            "PASSES_AIC_BH_5PCT"
        ]
        &
        (
            date_df[
                "POSITIVE_BETA"
            ]
            if
            REQUIRE_POSITIVE_BETA
            else
            True
        )
    )


    directional_rows.extend(
        date_df.to_dict(
            orient="records"
        )
    )


    input_audit_rows.append(
        {
            "FORMATION_DATE":
                formation_date,

            "BLOCK_TYPE":
                formation_row.BLOCK_TYPE,

            "N_INVESTABLE":
                len(
                    companies
                ),

            "N_UNORDERED_PAIRS":
                n_unordered_pairs,

            "EXPECTED_DIRECTIONAL_TESTS":
                expected_directional_tests,

            "ACTUAL_DIRECTIONAL_TESTS":
                len(
                    date_df
                ),

            "N_VALID_DIRECTIONAL_TESTS":
                int(
                    date_df[
                        "VALID_TEST"
                    ].sum()
                ),

            "N_AIC_BH_5PCT_DIRECTION_SURVIVORS":
                int(
                    date_df[
                        "PASSES_AIC_BH_5PCT"
                    ].sum()
                ),

            "N_BIC_BH_5PCT_DIRECTION_SURVIVORS":
                int(
                    date_df[
                        "PASSES_BIC_BH_5PCT"
                    ].sum()
                ),

            "N_FIXED1_BH_5PCT_DIRECTION_SURVIVORS":
                int(
                    date_df[
                        "PASSES_FIXED1_BH_5PCT"
                    ].sum()
                ),

            "MAX_FORMATION_DATE_USED":
                window[
                    "DATE"
                ].max(),

            "LOOKAHEAD_ROWS":
                int(
                    (
                        window[
                            "DATE"
                        ]
                        >
                        formation_date
                    ).sum()
                ),
        }
    )


directional = pd.DataFrame(
    directional_rows
)

input_audit = pd.DataFrame(
    input_audit_rows
)


# =============================================================================
# 5. CHOOSE THE BETTER DIRECTION FOR EACH UNORDERED PAIR
# =============================================================================

best_direction_rows = []


for (
    formation_date,
    pair_id
), grp in directional.groupby(
    [
        "FORMATION_DATE",
        "PAIR_ID",
    ],
    sort=True
):
    if len(
        grp
    ) != 2:
        raise RuntimeError(
            f"{formation_date.date()} {pair_id}: expected exactly two directions; "
            f"found {len(grp)}."
        )


    eligible = grp[
        grp[
            "DIRECTION_ELIGIBLE_PRIMARY"
        ]
    ].copy()


    if eligible.empty:
        # Still save the better AIC direction descriptively, but mark the pair
        # ineligible. This makes the audit transparent.
        candidates = grp[
            grp[
                "VALID_TEST"
            ]
        ].copy()

        if candidates.empty:
            chosen = grp.sort_values(
                "DIRECTION"
            ).iloc[
                0
            ]

            pair_eligible = False

        else:
            chosen = candidates.sort_values(
                [
                    "AIC_PVALUE_BH",
                    "AIC_PVALUE_RAW",
                    "AIC_TEST_STAT",
                    "DIRECTION",
                ],
                ascending=[
                    True,
                    True,
                    True,
                    True,
                ]
            ).iloc[
                0
            ]

            pair_eligible = False

    else:
        chosen = eligible.sort_values(
            [
                "AIC_PVALUE_BH",
                "AIC_PVALUE_RAW",
                "AIC_TEST_STAT",
                "DIRECTION",
            ],
            ascending=[
                True,
                True,
                True,
                True,
            ]
        ).iloc[
            0
        ]

        pair_eligible = True


    reverse = grp[
        grp[
            "DIRECTION"
        ].ne(
            chosen[
                "DIRECTION"
            ]
        )
    ].iloc[
        0
    ]


    best_direction_rows.append(
        {
            "FORMATION_DATE":
                chosen[
                    "FORMATION_DATE"
                ],

            "BLOCK_TYPE":
                chosen[
                    "BLOCK_TYPE"
                ],

            "FORMATION_START":
                chosen[
                    "FORMATION_START"
                ],

            "TRADING_START":
                chosen[
                    "TRADING_START"
                ],

            "TRADING_END":
                chosen[
                    "TRADING_END"
                ],

            "PAIR_ID":
                pair_id,

            "UNORDERED_COMPANY_1":
                chosen[
                    "UNORDERED_COMPANY_1"
                ],

            "UNORDERED_COMPANY_2":
                chosen[
                    "UNORDERED_COMPANY_2"
                ],

            "CHOSEN_DIRECTION":
                chosen[
                    "DIRECTION"
                ],

            "COMPANY_A":
                chosen[
                    "DEPENDENT_COMPANY"
                ],

            "COMPANY_B":
                chosen[
                    "EXPLANATORY_COMPANY"
                ],

            "ALPHA":
                chosen[
                    "ALPHA"
                ],

            "BETA":
                chosen[
                    "BETA"
                ],

            "COMMON_OBSERVATIONS":
                chosen[
                    "COMMON_OBSERVATIONS"
                ],

            "RESIDUAL_MEAN":
                chosen[
                    "RESIDUAL_MEAN"
                ],

            "RESIDUAL_STD_DDOF1":
                chosen[
                    "RESIDUAL_STD_DDOF1"
                ],

            "COINT_T_STAT":
                chosen[
                    "AIC_TEST_STAT"
                ],

            "COINT_PVALUE_RAW":
                chosen[
                    "AIC_PVALUE_RAW"
                ],

            "COINT_PVALUE_BH":
                chosen[
                    "AIC_PVALUE_BH"
                ],

            "CRITICAL_1PCT":
                chosen[
                    "AIC_CRITICAL_1PCT"
                ],

            "CRITICAL_5PCT":
                chosen[
                    "AIC_CRITICAL_5PCT"
                ],

            "CRITICAL_10PCT":
                chosen[
                    "AIC_CRITICAL_10PCT"
                ],

            "PAIR_ELIGIBLE_PRIMARY":
                pair_eligible,

            "CHOSEN_AIC_BH_PASS":
                bool(
                    chosen[
                        "PASSES_AIC_BH_5PCT"
                    ]
                ),

            "CHOSEN_BIC_BH_PASS":
                bool(
                    chosen[
                        "PASSES_BIC_BH_5PCT"
                    ]
                ),

            "CHOSEN_FIXED1_BH_PASS":
                bool(
                    chosen[
                        "PASSES_FIXED1_BH_5PCT"
                    ]
                ),

            "CHOSEN_BIC_PVALUE_RAW":
                chosen[
                    "BIC_PVALUE_RAW"
                ],

            "CHOSEN_BIC_PVALUE_BH":
                chosen[
                    "BIC_PVALUE_BH"
                ],

            "CHOSEN_FIXED1_PVALUE_RAW":
                chosen[
                    "FIXED1_PVALUE_RAW"
                ],

            "CHOSEN_FIXED1_PVALUE_BH":
                chosen[
                    "FIXED1_PVALUE_BH"
                ],

            "REVERSE_DIRECTION":
                reverse[
                    "DIRECTION"
                ],

            "REVERSE_AIC_PVALUE_RAW":
                reverse[
                    "AIC_PVALUE_RAW"
                ],

            "REVERSE_AIC_PVALUE_BH":
                reverse[
                    "AIC_PVALUE_BH"
                ],

            "REVERSE_AIC_BH_PASS":
                bool(
                    reverse[
                        "PASSES_AIC_BH_5PCT"
                    ]
                ),

            "DIRECTIONS_PASSING_PRIMARY_RULE":
                int(
                    grp[
                        "DIRECTION_ELIGIBLE_PRIMARY"
                    ].sum()
                ),
        }
    )


best_direction = pd.DataFrame(
    best_direction_rows
)


# =============================================================================
# 6. RANK ELIGIBLE UNORDERED PAIRS AND SELECT UP TO 4 NON-OVERLAPPING
# =============================================================================

best_direction[
    "EVIDENCE_RANK"
] = np.nan


for formation_date, grp in best_direction.groupby(
    "FORMATION_DATE",
    sort=True
):
    eligible_idx = grp[
        grp[
            "PAIR_ELIGIBLE_PRIMARY"
        ]
    ].index


    if len(
        eligible_idx
    ):
        ordered_idx = (
            best_direction.loc[
                eligible_idx
            ]
            .sort_values(
                [
                    "COINT_PVALUE_BH",
                    "COINT_PVALUE_RAW",
                    "COINT_T_STAT",
                    "PAIR_ID",
                ],
                ascending=[
                    True,
                    True,
                    True,
                    True,
                ]
            )
            .index
        )


        for rank, idx in enumerate(
            ordered_idx,
            start=1
        ):
            best_direction.loc[
                idx,
                "EVIDENCE_RANK",
            ] = rank


selected_rows = []


for formation_date, grp in best_direction.groupby(
    "FORMATION_DATE",
    sort=True
):
    eligible = grp[
        grp[
            "PAIR_ELIGIBLE_PRIMARY"
        ]
    ].copy()


    selected_ids, max_feasible = exact_disjoint_selection(
        eligible,
        MAX_SELECTED_PAIRS
    )


    picked = grp[
        grp[
            "PAIR_ID"
        ].isin(
            selected_ids
        )
    ].copy()


    picked = picked.sort_values(
        [
            "EVIDENCE_RANK",
            "PAIR_ID",
        ]
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
                    row.FORMATION_DATE,

                "BLOCK_TYPE":
                    row.BLOCK_TYPE,

                "SELECTED_PAIR_NUMBER":
                    selected_number,

                "PAIR_ID":
                    row.PAIR_ID,

                # IMPORTANT:
                # A/B now mean the chosen EG direction, not alphabetical order.
                "COMPANY_A":
                    row.COMPANY_A,

                "COMPANY_B":
                    row.COMPANY_B,

                "CHOSEN_DIRECTION":
                    row.CHOSEN_DIRECTION,

                "FORMATION_START":
                    row.FORMATION_START,

                "TRADING_START":
                    row.TRADING_START,

                "TRADING_END":
                    row.TRADING_END,

                "COMMON_OBSERVATIONS":
                    row.COMMON_OBSERVATIONS,

                "ALPHA":
                    row.ALPHA,

                "BETA":
                    row.BETA,

                "COINT_T_STAT":
                    row.COINT_T_STAT,

                "COINT_PVALUE_RAW":
                    row.COINT_PVALUE_RAW,

                "COINT_PVALUE_BH":
                    row.COINT_PVALUE_BH,

                "CRITICAL_1PCT":
                    row.CRITICAL_1PCT,

                "CRITICAL_5PCT":
                    row.CRITICAL_5PCT,

                "CRITICAL_10PCT":
                    row.CRITICAL_10PCT,

                "RESIDUAL_MEAN":
                    row.RESIDUAL_MEAN,

                "RESIDUAL_STD_DDOF1":
                    row.RESIDUAL_STD_DDOF1,

                "EVIDENCE_RANK":
                    int(
                        row.EVIDENCE_RANK
                    ),

                "DIRECTIONS_PASSING_PRIMARY_RULE":
                    row.DIRECTIONS_PASSING_PRIMARY_RULE,

                "REVERSE_DIRECTION":
                    row.REVERSE_DIRECTION,

                "REVERSE_AIC_PVALUE_RAW":
                    row.REVERSE_AIC_PVALUE_RAW,

                "REVERSE_AIC_PVALUE_BH":
                    row.REVERSE_AIC_PVALUE_BH,

                "BIC_BH_PASS_SENSITIVITY":
                    row.CHOSEN_BIC_BH_PASS,

                "FIXED1_BH_PASS_SENSITIVITY":
                    row.CHOSEN_FIXED1_BH_PASS,

                "BIC_PVALUE_BH_SENSITIVITY":
                    row.CHOSEN_BIC_PVALUE_BH,

                "FIXED1_PVALUE_BH_SENSITIVITY":
                    row.CHOSEN_FIXED1_PVALUE_BH,

                "MAX_FEASIBLE_DISJOINT_PAIRS_CAPPED_AT_4":
                    max_feasible,
            }
        )


selected = pd.DataFrame(
    selected_rows
)


if selected.empty:
    selected = pd.DataFrame(
        columns=[
            "FORMATION_DATE",
            "BLOCK_TYPE",
            "SELECTED_PAIR_NUMBER",
            "PAIR_ID",
            "COMPANY_A",
            "COMPANY_B",
            "CHOSEN_DIRECTION",
            "FORMATION_START",
            "TRADING_START",
            "TRADING_END",
            "COMMON_OBSERVATIONS",
            "ALPHA",
            "BETA",
            "COINT_T_STAT",
            "COINT_PVALUE_RAW",
            "COINT_PVALUE_BH",
            "CRITICAL_1PCT",
            "CRITICAL_5PCT",
            "CRITICAL_10PCT",
            "RESIDUAL_MEAN",
            "RESIDUAL_STD_DDOF1",
            "EVIDENCE_RANK",
            "DIRECTIONS_PASSING_PRIMARY_RULE",
            "REVERSE_DIRECTION",
            "REVERSE_AIC_PVALUE_RAW",
            "REVERSE_AIC_PVALUE_BH",
            "BIC_BH_PASS_SENSITIVITY",
            "FIXED1_BH_PASS_SENSITIVITY",
            "BIC_PVALUE_BH_SENSITIVITY",
            "FIXED1_PVALUE_BH_SENSITIVITY",
            "MAX_FEASIBLE_DISJOINT_PAIRS_CAPPED_AT_4",
        ]
    )


# =============================================================================
# 7. SAVE FORMATION GAP PATHS FOR SELECTED DIRECTIONS
# =============================================================================

path_frames = []


for row in selected.itertuples(
    index=False
):
    key = (
        pd.Timestamp(
            row.FORMATION_DATE
        ),
        row.PAIR_ID,
        row.CHOSEN_DIRECTION,
    )


    if key not in residual_cache:
        raise RuntimeError(
            f"Missing residual path for selected direction: {key}"
        )


    path = residual_cache[
        key
    ].copy()


    path[
        "SELECTED_PAIR_NUMBER"
    ] = int(
        row.SELECTED_PAIR_NUMBER
    )

    path[
        "ALPHA"
    ] = float(
        row.ALPHA
    )

    path[
        "BETA"
    ] = float(
        row.BETA
    )

    path[
        "FORMATION_RESIDUAL_MEAN"
    ] = float(
        row.RESIDUAL_MEAN
    )

    path[
        "FORMATION_RESIDUAL_STD"
    ] = float(
        row.RESIDUAL_STD_DDOF1
    )


    path_frames.append(
        path
    )


if path_frames:
    selected_paths = pd.concat(
        path_frames,
        ignore_index=True
    )

    selected_paths = selected_paths[
        [
            "FORMATION_DATE",
            "BLOCK_TYPE",
            "SELECTED_PAIR_NUMBER",
            "PAIR_ID",
            "DIRECTION",
            "COMPANY_A",
            "COMPANY_B",
            "DATE",
            "LOG_A",
            "LOG_B",
            "ALPHA",
            "BETA",
            "RESIDUAL_SPREAD",
            "FORMATION_RESIDUAL_MEAN",
            "FORMATION_RESIDUAL_STD",
        ]
    ].sort_values(
        [
            "FORMATION_DATE",
            "SELECTED_PAIR_NUMBER",
            "DATE",
        ]
    )

else:
    selected_paths = pd.DataFrame()


# =============================================================================
# 8. LAG-CHOICE SENSITIVITY FOR SELECTED PAIRS
# =============================================================================
#
# Plain English:
# AIC is our main way of deciding how much recent history the EG test allows.
# The assignment asks us to check whether the conclusion changes if we use
# another reasonable rule. So selected pairs are also shown under BIC and
# fixed lag=1. This is a check, NOT a second strategy.
# =============================================================================

if selected.empty:
    lag_sensitivity = pd.DataFrame()

else:
    lag_sensitivity = selected[
        [
            "FORMATION_DATE",
            "BLOCK_TYPE",
            "PAIR_ID",
            "CHOSEN_DIRECTION",
            "COMPANY_A",
            "COMPANY_B",
            "COINT_PVALUE_RAW",
            "COINT_PVALUE_BH",
            "BIC_PVALUE_BH_SENSITIVITY",
            "FIXED1_PVALUE_BH_SENSITIVITY",
            "BIC_BH_PASS_SENSITIVITY",
            "FIXED1_BH_PASS_SENSITIVITY",
        ]
    ].copy()


    lag_sensitivity[
        "AIC_BH_PASS_PRIMARY"
    ] = (
        lag_sensitivity[
            "COINT_PVALUE_BH"
        ]
        <=
        FDR_LEVEL
    )


    lag_sensitivity[
        "ALL_THREE_LAG_CHOICES_PASS"
    ] = (
        lag_sensitivity[
            "AIC_BH_PASS_PRIMARY"
        ]
        &
        lag_sensitivity[
            "BIC_BH_PASS_SENSITIVITY"
        ].astype(
            bool
        )
        &
        lag_sensitivity[
            "FIXED1_BH_PASS_SENSITIVITY"
        ].astype(
            bool
        )
    )


# =============================================================================
# 9. AUDIT
# =============================================================================

audit_rows = []


for formation_date, formation_row in formation_table.set_index(
    "FORMATION_DATE"
).iterrows():
    directional_here = directional[
        directional[
            "FORMATION_DATE"
        ].eq(
            formation_date
        )
    ]

    best_here = best_direction[
        best_direction[
            "FORMATION_DATE"
        ].eq(
            formation_date
        )
    ]

    selected_here = selected[
        selected[
            "FORMATION_DATE"
        ].eq(
            formation_date
        )
    ]


    expected_unordered = (
        int(
            formation_row[
                "N_FINAL_INVESTABLE"
            ]
        )
        *
        (
            int(
                formation_row[
                    "N_FINAL_INVESTABLE"
                ]
            )
            -
            1
        )
        //
        2
    )


    expected_directional = (
        2
        *
        expected_unordered
    )


    company_counts = pd.concat(
        [
            selected_here[
                "COMPANY_A"
            ],
            selected_here[
                "COMPANY_B"
            ],
        ],
        ignore_index=True
    ).value_counts()


    overlap_failures = int(
        (
            company_counts
            >
            1
        ).sum()
    )


    selected_bad_q = (
        int(
            (
                selected_here[
                    "COINT_PVALUE_BH"
                ]
                >
                FDR_LEVEL
            ).sum()
        )
        if
        len(
            selected_here
        )
        else
        0
    )


    selected_bad_beta = (
        int(
            (
                selected_here[
                    "BETA"
                ]
                <=
                0
            ).sum()
        )
        if
        len(
            selected_here
        )
        else
        0
    )


    eligible_here = best_here[
        best_here[
            "PAIR_ELIGIBLE_PRIMARY"
        ]
    ]


    _, max_feasible = exact_disjoint_selection(
        eligible_here,
        MAX_SELECTED_PAIRS
    )


    status = (
        "PASS"
        if
        len(
            directional_here
        )
        ==
        expected_directional
        and
        directional_here.groupby(
            "PAIR_ID"
        ).size().eq(
            2
        ).all()
        and
        len(
            best_here
        )
        ==
        expected_unordered
        and
        len(
            selected_here
        )
        ==
        max_feasible
        and
        len(
            selected_here
        )
        <=
        MAX_SELECTED_PAIRS
        and
        overlap_failures
        ==
        0
        and
        selected_bad_q
        ==
        0
        and
        selected_bad_beta
        ==
        0
        else
        "FAIL"
    )


    audit_rows.append(
        {
            "FORMATION_DATE":
                formation_date,

            "BLOCK_TYPE":
                formation_row[
                    "BLOCK_TYPE"
                ],

            "N_INVESTABLE":
                int(
                    formation_row[
                        "N_FINAL_INVESTABLE"
                    ]
                ),

            "EXPECTED_UNORDERED_PAIRS":
                expected_unordered,

            "EXPECTED_DIRECTIONAL_TESTS":
                expected_directional,

            "ACTUAL_DIRECTIONAL_TESTS":
                len(
                    directional_here
                ),

            "N_VALID_DIRECTIONAL_TESTS":
                int(
                    directional_here[
                        "VALID_TEST"
                    ].sum()
                ),

            "N_AIC_BH_5PCT_DIRECTION_SURVIVORS":
                int(
                    directional_here[
                        "PASSES_AIC_BH_5PCT"
                    ].sum()
                ),

            "N_UNORDERED_PAIRS_WITH_AT_LEAST_ONE_PRIMARY_DIRECTION":
                int(
                    best_here[
                        "PAIR_ELIGIBLE_PRIMARY"
                    ].sum()
                ),

            "N_SELECTED":
                len(
                    selected_here
                ),

            "MAX_FEASIBLE_SELECTED_UP_TO_4":
                max_feasible,

            "OVERLAPPING_SELECTED_COMPANIES":
                overlap_failures,

            "SELECTED_FAILING_AIC_BH_5PCT":
                selected_bad_q,

            "SELECTED_NONPOSITIVE_BETA":
                selected_bad_beta,

            "STATUS":
                status,
        }
    )


audit = pd.DataFrame(
    audit_rows
)


if (
    audit[
        "STATUS"
    ]
    !=
    "PASS"
).any():
    raise RuntimeError(
        "Bidirectional EG pair-selection audit failed:\n\n"
        +
        audit[
            audit[
                "STATUS"
            ]
            !=
            "PASS"
        ].to_string(
            index=False
        )
    )


summary = pd.DataFrame(
    [
        {
            "CHECK":
                "Overall status",
            "VALUE":
                "PASS",
        },
        {
            "CHECK":
                "Formation dates processed",
            "VALUE":
                formation_table[
                    "FORMATION_DATE"
                ].nunique(),
        },
        {
            "CHECK":
                "Unordered pair-periods examined",
            "VALUE":
                len(
                    best_direction
                ),
        },
        {
            "CHECK":
                "Directional EG tests attempted",
            "VALUE":
                len(
                    directional
                ),
        },
        {
            "CHECK":
                "Valid directional EG tests",
            "VALUE":
                int(
                    directional[
                        "VALID_TEST"
                    ].sum()
                ),
        },
        {
            "CHECK":
                "AIC directional tests surviving corrected 5% rule",
            "VALUE":
                int(
                    directional[
                        "PASSES_AIC_BH_5PCT"
                    ].sum()
                ),
        },
        {
            "CHECK":
                "Unordered pair-periods with at least one valid direction",
            "VALUE":
                int(
                    best_direction[
                        "PAIR_ELIGIBLE_PRIMARY"
                    ].sum()
                ),
        },
        {
            "CHECK":
                "Selected pair-periods",
            "VALUE":
                len(
                    selected
                ),
        },
        {
            "CHECK":
                "Selected pairs passing BIC sensitivity",
            "VALUE":
                (
                    int(
                        selected[
                            "BIC_BH_PASS_SENSITIVITY"
                        ].astype(
                            bool
                        ).sum()
                    )
                    if
                    len(
                        selected
                    )
                    else
                    0
                ),
        },
        {
            "CHECK":
                "Selected pairs passing fixed-lag-1 sensitivity",
            "VALUE":
                (
                    int(
                        selected[
                            "FIXED1_BH_PASS_SENSITIVITY"
                        ].astype(
                            bool
                        ).sum()
                    )
                    if
                    len(
                        selected
                    )
                    else
                    0
                ),
        },
        {
            "CHECK":
                "Formation dates with zero selected pairs",
            "VALUE":
                int(
                    (
                        audit[
                            "N_SELECTED"
                        ]
                        ==
                        0
                    ).sum()
                ),
        },
        {
            "CHECK":
                "Maximum pairs selected on one formation date",
            "VALUE":
                int(
                    audit[
                        "N_SELECTED"
                    ].max()
                ),
        },
        {
            "CHECK":
                "Look-ahead failures",
            "VALUE":
                int(
                    input_audit[
                        "LOOKAHEAD_ROWS"
                    ].sum()
                ),
        },
        {
            "CHECK":
                "Trading signals generated",
            "VALUE":
                "NO",
        },
        {
            "CHECK":
                "P&L calculated",
            "VALUE":
                "NO",
        },
    ]
)


# =============================================================================
# 10. CONFIG
# =============================================================================

config = pd.DataFrame(
    [
        [
            "METHOD",
            "ENGLE_GRANGER_TWO_STEP_BIDIRECTIONAL",
        ],
        [
            "FORMATION_DATA",
            "AUDITED_LOG_TOTAL_RETURN_INDEX",
        ],
        [
            "ASYMMETRY_RULE",
            "TEST_BOTH_DIRECTIONS; COUNT_BOTH_IN_MULTIPLE_TESTING; KEEP_STRONGER_QUALIFYING_DIRECTION_PER_UNORDERED_PAIR",
        ],
        [
            "PRIMARY_LAG_RULE",
            "AIC",
        ],
        [
            "LAG_SENSITIVITY_1",
            "BIC",
        ],
        [
            "LAG_SENSITIVITY_2",
            "FIXED_LAG_1",
        ],
        [
            "EG_CRITICAL_VALUES",
            "STATSMODELS_COINT_MACKINNON",
        ],
        [
            "MULTIPLE_TESTING",
            "BENJAMINI_HOCHBERG_ACROSS_ALL_DIRECTIONAL_TESTS_ON_EACH_FORMATION_DATE",
        ],
        [
            "FDR_LEVEL",
            str(
                FDR_LEVEL
            ),
        ],
        [
            "POSITIVE_BETA_REQUIRED",
            str(
                REQUIRE_POSITIVE_BETA
            ),
        ],
        [
            "MAX_SELECTED_PAIRS",
            str(
                MAX_SELECTED_PAIRS
            ),
        ],
        [
            "PAIR_OVERLAP",
            "NOT_ALLOWED",
        ],
        [
            "SIGNALS",
            "NOT_GENERATED",
        ],
        [
            "PNL",
            "NOT_CALCULATED",
        ],
    ],
    columns=[
        "PARAMETER",
        "VALUE",
    ]
)


# =============================================================================
# 11. SAVE
# =============================================================================

# Standard filenames are intentionally replaced so the later backtest reads the
# corrected bidirectional EG selection instead of the old alphabetical version.

config.to_csv(
    PAIR_SELECTION_DIR
    / "00_ENGLE_GRANGER_METHOD_CONFIG.csv",
    index=False
)

input_audit.to_csv(
    PAIR_SELECTION_DIR
    / "01_FORMATION_INPUT_AUDIT.csv",
    index=False,
    date_format="%Y-%m-%d"
)

directional.to_csv(
    PAIR_SELECTION_DIR
    / "02_ALL_DIRECTIONAL_ENGLE_GRANGER_TESTS.csv",
    index=False,
    date_format="%Y-%m-%d"
)

best_direction.to_csv(
    PAIR_SELECTION_DIR
    / "02B_BEST_DIRECTION_PER_UNORDERED_PAIR.csv",
    index=False,
    date_format="%Y-%m-%d"
)

selected.to_csv(
    PAIR_SELECTION_DIR
    / "03_SELECTED_ENGLE_GRANGER_PAIRS.csv",
    index=False,
    date_format="%Y-%m-%d"
)

selected_paths.to_csv(
    PAIR_SELECTION_DIR
    / "04_SELECTED_PAIR_FORMATION_RESIDUAL_PATHS.csv",
    index=False,
    date_format="%Y-%m-%d"
)

lag_sensitivity.to_csv(
    PAIR_SELECTION_DIR
    / "05_SELECTED_PAIR_LAG_SENSITIVITY.csv",
    index=False,
    date_format="%Y-%m-%d"
)

audit.to_csv(
    AUDIT_DIR
    / "00_ENGLE_GRANGER_PAIR_SELECTION_AUDIT.csv",
    index=False,
    date_format="%Y-%m-%d"
)

summary.to_csv(
    AUDIT_DIR
    / "01_ENGLE_GRANGER_PAIR_SELECTION_SUMMARY.csv",
    index=False
)


# =============================================================================
# 12. CONSOLE REPORT
# =============================================================================

print(
    "\n"
    +
    "="
    *
    116
)

print(
    "BIDIRECTIONAL ENGLE-GRANGER PAIR SELECTION COMPLETE — AUDIT PASS"
)

print(
    "="
    *
    116
)


print(
    "\nSUMMARY"
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\nSELECTED PAIRS"
)


if selected.empty:
    print(
        "No pairs selected."
    )

else:
    print(
        selected[
            [
                "FORMATION_DATE",
                "BLOCK_TYPE",
                "PAIR_ID",
                "CHOSEN_DIRECTION",
                "COMPANY_A",
                "COMPANY_B",
                "BETA",
                "COINT_PVALUE_BH",
                "BIC_BH_PASS_SENSITIVITY",
                "FIXED1_BH_PASS_SENSITIVITY",
            ]
        ].to_string(
            index=False
        )
    )


print(
    "\nOutput folder:"
)

print(
    PAIR_SELECTION_DIR
)


print(
    "\nUpload these four files next:"
)

for path in [
    PAIR_SELECTION_DIR
    / "03_SELECTED_ENGLE_GRANGER_PAIRS.csv",

    PAIR_SELECTION_DIR
    / "05_SELECTED_PAIR_LAG_SENSITIVITY.csv",

    AUDIT_DIR
    / "00_ENGLE_GRANGER_PAIR_SELECTION_AUDIT.csv",

    AUDIT_DIR
    / "01_ENGLE_GRANGER_PAIR_SELECTION_SUMMARY.csv",
]:
    print(
        path
    )


print(
    "\nIMPORTANT: do not run the old EG backtest yet."
)

print(
    "Its hard-coded expected selected-pair count must be updated after we see this new result."
)
