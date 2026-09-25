import pandas as pd
import numpy as np
from pathlib import Path
import os


# =============================================================================
# NIFTY PHARMA — FORMATION-DATE INVESTABILITY DIAGNOSTICS
# =============================================================================
#
# PURPOSE
# -------
# Construct the point-in-time candidate universe at each six-month formation
# date using ONLY information available on or before that formation date.
#
# THIS STAGE DOES:
#   - 12-month formation windows
#   - 6-month trading blocks
#   - point-in-time NIFTY Pharma membership at formation date
#   - price-history coverage
#   - maximum consecutive missing-market-session gap
#   - structural-segment continuity
#   - price-floor diagnostics
#   - formation-date tradability
#   - formation-window liquidity diagnostics using RAW NSE TOTAL_TRADED_VALUE
#   - development/OOS block labelling
#
# THIS STAGE DOES NOT:
#   - choose a final liquidity threshold
#   - reconstruct historical F&O eligibility
#   - select pairs
#   - run SSD
#   - run Engle-Granger
#   - inspect strategy P&L
#
# Final investability remains unresolved until liquidity and F&O rules are
# frozen.
# =============================================================================


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(
    os.environ.get("PAIR_TRADING_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()

POINT_IN_TIME_FILE = (
    PROJECT_ROOT
    / "nse_pharma_point_in_time_universe"
    / "NIFTY_PHARMA_POINT_IN_TIME_FLAGGED_2016_2026.csv"
)

MEMBERSHIP_FILENAME = (
    "historical_nifty_pharma_FINAL_AUDITED.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "nse_pharma_formation_investability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. FROZEN DESIGN PARAMETERS
# =============================================================================

FORMATION_MONTHS = 12
TRADING_MONTHS = 6

# Non-liquidity rules proposed before looking at strategy P&L.
MIN_COVERAGE_PCT = 95.0
MAX_MISSING_GAP_SESSIONS = 5
MIN_MEDIAN_CLOSE = 50.0

# Final untouched OOS period.
OOS_START = pd.Timestamp("2024-08-01")
OOS_END   = pd.Timestamp("2026-07-31")

# First anchor that has a full 12 calendar months of data available.
FIRST_FORMATION_ANCHOR = pd.Timestamp("2017-01-31")

# Last six-month block that can be fully observed through 31-Jul-2026.
LAST_FORMATION_ANCHOR = pd.Timestamp("2026-01-31")


# =============================================================================
# 3. HELPERS
# =============================================================================

def parse_dates(series):
    try:
        return pd.to_datetime(
            series,
            format="mixed",
            errors="coerce"
        )
    except TypeError:
        return pd.to_datetime(
            series,
            errors="coerce"
        )


def clean_upper(series):
    return (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )


def normalize_membership_symbol(symbol):
    if pd.isna(symbol):
        return pd.NA

    s = str(symbol).strip().upper()

    # Known audited membership spelling mismatch.
    if s == "AJANTAPHARM":
        return "AJANTPHARM"

    return s


def stable_company_id(symbol):
    if pd.isna(symbol):
        return pd.NA

    s = str(symbol).strip().upper()

    if s in {
        "CADILAHC",
        "ZYDUSLIFE",
    }:
        return "ZYDUS"

    if s == "AJANTAPHARM":
        return "AJANTPHARM"

    return s


def locate_membership_file():
    direct = (
        PROJECT_ROOT
        /
        MEMBERSHIP_FILENAME
    )

    if direct.exists():
        return direct

    matches = list(
        PROJECT_ROOT.rglob(
            MEMBERSHIP_FILENAME
        )
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            "\nCould not locate finalized audited membership file:\n"
            f"{MEMBERSHIP_FILENAME}\n"
            f"under {PROJECT_ROOT}"
        )

    raise RuntimeError(
        "\nMore than one finalized membership file was found.\n"
        "Refusing to guess which is authoritative:\n\n"
        +
        "\n".join(
            str(x)
            for x in matches
        )
    )


def last_market_session_on_or_before(
    market_dates,
    calendar_date
):
    eligible = market_dates[
        market_dates
        <= calendar_date
    ]

    if len(eligible) == 0:
        return pd.NaT

    return eligible.max()


def first_market_session_after(
    market_dates,
    date
):
    eligible = market_dates[
        market_dates
        > date
    ]

    if len(eligible) == 0:
        return pd.NaT

    return eligible.min()


def max_consecutive_missing_sessions(
    expected_dates,
    actual_dates
):
    """
    Missing-run length measured in MARKET SESSIONS, not calendar days.
    """

    expected = pd.Index(
        expected_dates
    )

    actual = set(
        pd.Timestamp(x)
        for x in actual_dates
    )

    if len(expected) == 0:
        return 0

    longest = 0
    current = 0

    for date in expected:

        if pd.Timestamp(date) in actual:
            current = 0

        else:
            current += 1
            longest = max(
                longest,
                current
            )

    return int(
        longest
    )


def membership_at_date(
    membership,
    company_id,
    date
):
    sub = membership[
        membership[
            "COMPANY_ID"
        ].eq(
            company_id
        )
        &
        membership[
            "START_DATE"
        ].le(
            date
        )
        &
        membership[
            "END_DATE"
        ].ge(
            date
        )
    ]

    if len(sub) == 0:
        return (
            False,
            pd.NA,
            pd.NA,
            pd.NaT,
            pd.NaT
        )

    if len(sub) > 1:
        raise RuntimeError(
            f"Multiple membership intervals matched "
            f"{company_id} on {date.date()}."
        )

    row = sub.iloc[0]

    return (
        True,
        row[
            "MEMBERSHIP_SYMBOL"
        ],
        row[
            "COMPANY"
        ],
        row[
            "START_DATE"
        ],
        row[
            "END_DATE"
        ],
    )


def percentile_or_nan(
    series,
    q
):
    x = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(x) == 0:
        return np.nan

    return float(
        x.quantile(
            q
        )
    )


# =============================================================================
# 4. LOAD POINT-IN-TIME DATA
# =============================================================================

print("=" * 110)
print("FORMATION-DATE INVESTABILITY DIAGNOSTICS")
print("=" * 110)

if not POINT_IN_TIME_FILE.exists():
    raise FileNotFoundError(
        f"Point-in-time file not found:\n{POINT_IN_TIME_FILE}"
    )

data = pd.read_csv(
    POINT_IN_TIME_FILE,
    low_memory=False
)

data.columns = (
    data.columns
    .astype(str)
    .str.replace(
        "\ufeff",
        "",
        regex=False
    )
    .str.strip()
    .str.upper()
)

required_columns = {
    "DATE",
    "SYMBOL",
    "COMPANY_ID",
    "CLOSE",
    "TOTAL_TRADED_VALUE",
    "SEGMENT_ID",
    "STRUCTURAL_BREAK_FLAG",
}

missing_columns = (
    required_columns
    -
    set(
        data.columns
    )
)

if missing_columns:
    raise ValueError(
        "Point-in-time dataset is missing required columns:\n"
        f"{sorted(missing_columns)}"
    )


data["DATE"] = parse_dates(
    data["DATE"]
)

data["SYMBOL"] = clean_upper(
    data["SYMBOL"]
)

data["COMPANY_ID"] = clean_upper(
    data["COMPANY_ID"]
)

data["SEGMENT_ID"] = clean_upper(
    data["SEGMENT_ID"]
)

data["CLOSE"] = pd.to_numeric(
    data["CLOSE"],
    errors="coerce"
)

data["TOTAL_TRADED_VALUE"] = pd.to_numeric(
    data["TOTAL_TRADED_VALUE"],
    errors="coerce"
)


if data[
    "DATE"
].isna().any():
    raise ValueError(
        "Invalid DATE values exist."
    )


if data[
    [
        "DATE",
        "COMPANY_ID",
    ]
].duplicated().any():
    raise RuntimeError(
        "Duplicate DATE + COMPANY_ID rows exist."
    )


if data[
    "CLOSE"
].isna().any():
    raise RuntimeError(
        "Missing CLOSE exists in point-in-time price rows."
    )


if (
    data[
        "CLOSE"
    ] <= 0
).any():
    raise RuntimeError(
        "Non-positive CLOSE exists."
    )


# =============================================================================
# 5. LOAD HISTORICAL MEMBERSHIP
# =============================================================================

MEMBERSHIP_FILE = (
    locate_membership_file()
)

membership = pd.read_csv(
    MEMBERSHIP_FILE,
    low_memory=False
)

membership.columns = (
    membership.columns
    .astype(str)
    .str.replace(
        "\ufeff",
        "",
        regex=False
    )
    .str.strip()
    .str.upper()
)

required_membership = {
    "SYMBOL",
    "COMPANY",
    "START_DATE",
    "END_DATE",
}

missing_membership = (
    required_membership
    -
    set(
        membership.columns
    )
)

if missing_membership:
    raise ValueError(
        "Membership file missing columns:\n"
        f"{sorted(missing_membership)}"
    )


membership["START_DATE"] = parse_dates(
    membership["START_DATE"]
)

membership["END_DATE"] = parse_dates(
    membership["END_DATE"]
)

membership[
    "MEMBERSHIP_SYMBOL_SOURCE"
] = clean_upper(
    membership[
        "SYMBOL"
    ]
)

membership[
    "MEMBERSHIP_SYMBOL"
] = (
    membership[
        "MEMBERSHIP_SYMBOL_SOURCE"
    ]
    .map(
        normalize_membership_symbol
    )
    .astype(
        "string"
    )
)

membership[
    "COMPANY_ID"
] = (
    membership[
        "MEMBERSHIP_SYMBOL"
    ]
    .map(
        stable_company_id
    )
    .astype(
        "string"
    )
)


if membership[
    [
        "START_DATE",
        "END_DATE",
    ]
].isna().any().any():
    raise ValueError(
        "Invalid membership dates."
    )


# =============================================================================
# 6. BUILD SIX-MONTH FORMATION SCHEDULE
# =============================================================================

market_dates = pd.Series(
    sorted(
        data[
            "DATE"
        ].unique()
    )
)


anchors = []

year = FIRST_FORMATION_ANCHOR.year

while True:

    for month, day in [
        (1, 31),
        (7, 31),
    ]:

        anchor = pd.Timestamp(
            year=year,
            month=month,
            day=day
        )

        if (
            anchor
            <
            FIRST_FORMATION_ANCHOR
        ):
            continue

        if (
            anchor
            >
            LAST_FORMATION_ANCHOR
        ):
            continue

        anchors.append(
            anchor
        )

    if pd.Timestamp(
        year=year,
        month=7,
        day=31
    ) >= LAST_FORMATION_ANCHOR:
        break

    year += 1


anchors = sorted(
    set(
        anchors
    )
)


schedule_records = []


for anchor in anchors:

    formation_date = (
        last_market_session_on_or_before(
            market_dates,
            anchor
        )
    )

    if pd.isna(
        formation_date
    ):
        continue

    formation_start_calendar = (
        anchor
        -
        pd.DateOffset(
            years=1
        )
        +
        pd.Timedelta(
            days=1
        )
    )

    trading_start = (
        first_market_session_after(
            market_dates,
            formation_date
        )
    )

    trading_end_anchor = (
        anchor
        +
        pd.DateOffset(
            months=TRADING_MONTHS
        )
    )

    trading_end = (
        last_market_session_on_or_before(
            market_dates,
            min(
                trading_end_anchor,
                OOS_END
                if trading_end_anchor > data[
                    "DATE"
                ].max()
                else trading_end_anchor
            )
        )
    )

    block_type = (
        "OOS"
        if (
            pd.notna(
                trading_start
            )
            and
            trading_start
            >=
            OOS_START
        )
        else "DEVELOPMENT"
    )

    schedule_records.append(
        {
            "FORMATION_ANCHOR":
                anchor,

            "FORMATION_START_CALENDAR":
                formation_start_calendar,

            "FORMATION_DATE":
                formation_date,

            "TRADING_START":
                trading_start,

            "TRADING_END_ANCHOR":
                trading_end_anchor,

            "TRADING_END":
                trading_end,

            "BLOCK_TYPE":
                block_type,
        }
    )


schedule = pd.DataFrame(
    schedule_records
)


if schedule.empty:
    raise RuntimeError(
        "No formation schedule was generated."
    )


# Ensure trading blocks do not overlap incorrectly.
schedule = schedule.sort_values(
    "FORMATION_ANCHOR"
).reset_index(
    drop=True
)


# =============================================================================
# 7. BUILD STOCK × FORMATION-DATE DIAGNOSTICS
# =============================================================================

company_ids = sorted(
    data[
        "COMPANY_ID"
    ].dropna().unique()
)


diagnostic_records = []


for _, block in schedule.iterrows():

    formation_anchor = block[
        "FORMATION_ANCHOR"
    ]

    formation_start = block[
        "FORMATION_START_CALENDAR"
    ]

    formation_date = block[
        "FORMATION_DATE"
    ]

    expected_dates = market_dates[
        (
            market_dates
            >=
            formation_start
        )
        &
        (
            market_dates
            <=
            formation_date
        )
    ]

    expected_sessions = len(
        expected_dates
    )

    if expected_sessions == 0:
        raise RuntimeError(
            f"No market sessions in formation window ending "
            f"{formation_date.date()}."
        )


    for company_id in company_ids:

        (
            member_flag,
            membership_symbol,
            membership_company,
            membership_start,
            membership_end,
        ) = membership_at_date(
            membership,
            company_id,
            formation_date
        )


        company_window = data[
            data[
                "COMPANY_ID"
            ].eq(
                company_id
            )
            &
            data[
                "DATE"
            ].between(
                formation_start,
                formation_date,
                inclusive="both"
            )
        ].copy()


        available_observations = len(
            company_window
        )

        coverage_pct = (
            100.0
            *
            available_observations
            /
            expected_sessions
        )


        max_missing_gap = (
            max_consecutive_missing_sessions(
                expected_dates,
                company_window[
                    "DATE"
                ].tolist()
            )
        )


        formation_day_rows = company_window[
            company_window[
                "DATE"
            ].eq(
                formation_date
            )
        ]


        formation_date_price_present = (
            len(
                formation_day_rows
            ) == 1
        )


        if len(
            formation_day_rows
        ) > 1:
            raise RuntimeError(
                f"Multiple formation-date rows for "
                f"{company_id} on {formation_date.date()}."
            )


        if formation_date_price_present:

            current_symbol = (
                formation_day_rows.iloc[0][
                    "SYMBOL"
                ]
            )

            formation_close = float(
                formation_day_rows.iloc[0][
                    "CLOSE"
                ]
            )

        else:

            current_symbol = pd.NA
            formation_close = np.nan


        n_segments = (
            company_window[
                "SEGMENT_ID"
            ]
            .dropna()
            .nunique()
        )


        segment_ids = (
            company_window[
                "SEGMENT_ID"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )


        single_segment_flag = (
            n_segments == 1
        )


        structural_break_rows = 0

        if (
            "STRUCTURAL_BREAK_FLAG"
            in company_window.columns
            and
            not company_window.empty
        ):

            sb = (
                company_window[
                    "STRUCTURAL_BREAK_FLAG"
                ]
                .astype(str)
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

            structural_break_rows = int(
                sb.sum()
            )


        median_close = (
            float(
                company_window[
                    "CLOSE"
                ].median()
            )
            if available_observations > 0
            else np.nan
        )


        min_close = (
            float(
                company_window[
                    "CLOSE"
                ].min()
            )
            if available_observations > 0
            else np.nan
        )


        median_traded_value = (
            float(
                company_window[
                    "TOTAL_TRADED_VALUE"
                ].median()
            )
            if available_observations > 0
            else np.nan
        )


        p10_traded_value = (
            percentile_or_nan(
                company_window[
                    "TOTAL_TRADED_VALUE"
                ],
                0.10
            )
        )


        p25_traded_value = (
            percentile_or_nan(
                company_window[
                    "TOTAL_TRADED_VALUE"
                ],
                0.25
            )
        )


        average_traded_value = (
            float(
                company_window[
                    "TOTAL_TRADED_VALUE"
                ].mean()
            )
            if available_observations > 0
            else np.nan
        )


        first_available_date = (
            company_window[
                "DATE"
            ].min()
            if available_observations > 0
            else pd.NaT
        )


        last_available_date = (
            company_window[
                "DATE"
            ].max()
            if available_observations > 0
            else pd.NaT
        )


        history_pass = (
            coverage_pct
            >=
            MIN_COVERAGE_PCT
        )


        gap_pass = (
            max_missing_gap
            <=
            MAX_MISSING_GAP_SESSIONS
        )


        segment_pass = (
            single_segment_flag
        )


        price_pass = (
            pd.notna(
                median_close
            )
            and
            median_close
            >=
            MIN_MEDIAN_CLOSE
        )


        formation_tradability_pass = (
            formation_date_price_present
        )


        pre_liquidity_eligible = bool(
            member_flag
            and
            history_pass
            and
            gap_pass
            and
            segment_pass
            and
            price_pass
            and
            formation_tradability_pass
        )


        exclusion_reasons = []

        if not member_flag:
            exclusion_reasons.append(
                "NOT_INDEX_MEMBER_AT_FORMATION"
            )

        if not history_pass:
            exclusion_reasons.append(
                "INSUFFICIENT_FORMATION_COVERAGE"
            )

        if not gap_pass:
            exclusion_reasons.append(
                "EXCESSIVE_MISSING_SESSION_GAP"
            )

        if not segment_pass:
            exclusion_reasons.append(
                "FORMATION_CROSSES_STRUCTURAL_SEGMENT"
            )

        if not price_pass:
            exclusion_reasons.append(
                "MEDIAN_PRICE_BELOW_FLOOR"
            )

        if not formation_tradability_pass:
            exclusion_reasons.append(
                "NO_EQ_PRICE_ON_FORMATION_DATE"
            )


        diagnostic_records.append(
            {
                "FORMATION_ANCHOR":
                    formation_anchor,

                "FORMATION_START":
                    formation_start,

                "FORMATION_DATE":
                    formation_date,

                "TRADING_START":
                    block[
                        "TRADING_START"
                    ],

                "TRADING_END":
                    block[
                        "TRADING_END"
                    ],

                "BLOCK_TYPE":
                    block[
                        "BLOCK_TYPE"
                    ],

                "COMPANY_ID":
                    company_id,

                "SYMBOL_ON_FORMATION_DATE":
                    current_symbol,

                "MEMBERSHIP_SYMBOL":
                    membership_symbol,

                "MEMBERSHIP_COMPANY":
                    membership_company,

                "MEMBERSHIP_START":
                    membership_start,

                "MEMBERSHIP_END":
                    membership_end,

                "IN_NIFTY_PHARMA_AT_FORMATION":
                    bool(
                        member_flag
                    ),

                "EXPECTED_MARKET_SESSIONS":
                    expected_sessions,

                "AVAILABLE_OBSERVATIONS":
                    available_observations,

                "COVERAGE_PCT":
                    coverage_pct,

                "MAX_CONSECUTIVE_MISSING_SESSIONS":
                    max_missing_gap,

                "FIRST_AVAILABLE_DATE_IN_WINDOW":
                    first_available_date,

                "LAST_AVAILABLE_DATE_IN_WINDOW":
                    last_available_date,

                "FORMATION_DATE_PRICE_PRESENT":
                    formation_date_price_present,

                "FORMATION_DATE_CLOSE":
                    formation_close,

                "N_SEGMENTS_IN_FORMATION":
                    n_segments,

                "SEGMENT_IDS_IN_FORMATION":
                    ";".join(
                        segment_ids
                    ),

                "STRUCTURAL_BREAK_ROWS_IN_FORMATION":
                    structural_break_rows,

                "SINGLE_SEGMENT_FLAG":
                    single_segment_flag,

                "MEDIAN_CLOSE":
                    median_close,

                "MIN_CLOSE":
                    min_close,

                "MEDIAN_DAILY_TRADED_VALUE":
                    median_traded_value,

                "MEDIAN_DAILY_TRADED_VALUE_CR":
                    (
                        median_traded_value
                        /
                        1e7
                        if pd.notna(
                            median_traded_value
                        )
                        else np.nan
                    ),

                "P10_DAILY_TRADED_VALUE":
                    p10_traded_value,

                "P10_DAILY_TRADED_VALUE_CR":
                    (
                        p10_traded_value
                        /
                        1e7
                        if pd.notna(
                            p10_traded_value
                        )
                        else np.nan
                    ),

                "P25_DAILY_TRADED_VALUE":
                    p25_traded_value,

                "P25_DAILY_TRADED_VALUE_CR":
                    (
                        p25_traded_value
                        /
                        1e7
                        if pd.notna(
                            p25_traded_value
                        )
                        else np.nan
                    ),

                "AVERAGE_DAILY_TRADED_VALUE":
                    average_traded_value,

                "AVERAGE_DAILY_TRADED_VALUE_CR":
                    (
                        average_traded_value
                        /
                        1e7
                        if pd.notna(
                            average_traded_value
                        )
                        else np.nan
                    ),

                "HISTORY_PASS":
                    bool(
                        history_pass
                    ),

                "GAP_PASS":
                    bool(
                        gap_pass
                    ),

                "SEGMENT_PASS":
                    bool(
                        segment_pass
                    ),

                "PRICE_PASS":
                    bool(
                        price_pass
                    ),

                "FORMATION_TRADABILITY_PASS":
                    bool(
                        formation_tradability_pass
                    ),

                "PRE_LIQUIDITY_ELIGIBLE":
                    pre_liquidity_eligible,

                # Deliberately unresolved at this stage.
                "LIQUIDITY_THRESHOLD_RUPEES":
                    np.nan,

                "LIQUIDITY_PASS":
                    pd.NA,

                "FNO_ELIGIBLE":
                    pd.NA,

                "FINAL_INVESTABLE":
                    pd.NA,

                "EXCLUSION_REASONS":
                    (
                        ";".join(
                            exclusion_reasons
                        )
                        if exclusion_reasons
                        else ""
                    ),
            }
        )


diagnostics = pd.DataFrame(
    diagnostic_records
)


# Nullable booleans for unresolved fields.
for col in [
    "LIQUIDITY_PASS",
    "FNO_ELIGIBLE",
    "FINAL_INVESTABLE",
]:

    diagnostics[
        col
    ] = diagnostics[
        col
    ].astype(
        "boolean"
    )


# =============================================================================
# 8. MEMBER-ONLY / PRE-LIQUIDITY SUBSETS
# =============================================================================

member_candidates = diagnostics[
    diagnostics[
        "IN_NIFTY_PHARMA_AT_FORMATION"
    ]
].copy()


pre_liquidity = diagnostics[
    diagnostics[
        "PRE_LIQUIDITY_ELIGIBLE"
    ]
].copy()


# =============================================================================
# 9. LIQUIDITY DISTRIBUTION — DEVELOPMENT ONLY
#
# IMPORTANT:
# We use DEVELOPMENT blocks only for choosing a fixed global liquidity cutoff.
# OOS liquidity diagnostics are not used to tune the threshold.
# =============================================================================

development_preliq = pre_liquidity[
    pre_liquidity[
        "BLOCK_TYPE"
    ].eq(
        "DEVELOPMENT"
    )
].copy()


liq_dist_records = []


for formation_date, grp in development_preliq.groupby(
    "FORMATION_DATE"
):

    x = (
        grp[
            "MEDIAN_DAILY_TRADED_VALUE"
        ]
        .dropna()
        .astype(float)
    )

    if len(x) == 0:
        continue

    liq_dist_records.append(
        {
            "FORMATION_DATE":
                formation_date,

            "N_PRE_LIQUIDITY_ELIGIBLE":
                len(
                    x
                ),

            "MIN_MEDIAN_DAILY_TRADED_VALUE":
                x.min(),

            "P10_MEDIAN_DAILY_TRADED_VALUE":
                x.quantile(
                    0.10
                ),

            "P20_MEDIAN_DAILY_TRADED_VALUE":
                x.quantile(
                    0.20
                ),

            "P25_MEDIAN_DAILY_TRADED_VALUE":
                x.quantile(
                    0.25
                ),

            "MEDIAN_OF_MEDIAN_DAILY_TRADED_VALUE":
                x.median(),

            "P75_MEDIAN_DAILY_TRADED_VALUE":
                x.quantile(
                    0.75
                ),

            "MAX_MEDIAN_DAILY_TRADED_VALUE":
                x.max(),
        }
    )


liquidity_distribution = pd.DataFrame(
    liq_dist_records
)


if not liquidity_distribution.empty:

    value_cols = [
        c
        for c in liquidity_distribution.columns
        if (
            c
            !=
            "FORMATION_DATE"
            and
            c
            !=
            "N_PRE_LIQUIDITY_ELIGIBLE"
        )
    ]

    for col in value_cols:

        liquidity_distribution[
            col
            +
            "_CR"
        ] = (
            liquidity_distribution[
                col
            ]
            /
            1e7
        )


# =============================================================================
# 10. FORMATION-DATE COUNTS
# =============================================================================

count_records = []


for formation_date, grp in diagnostics.groupby(
    "FORMATION_DATE"
):

    block_type = grp[
        "BLOCK_TYPE"
    ].iloc[0]

    count_records.append(
        {
            "FORMATION_DATE":
                formation_date,

            "BLOCK_TYPE":
                block_type,

            "TOTAL_COMPANY_IDENTITIES":
                grp[
                    "COMPANY_ID"
                ].nunique(),

            "INDEX_MEMBERS":
                int(
                    grp[
                        "IN_NIFTY_PHARMA_AT_FORMATION"
                    ].sum()
                ),

            "HISTORY_PASS_MEMBERS":
                int(
                    (
                        grp[
                            "IN_NIFTY_PHARMA_AT_FORMATION"
                        ]
                        &
                        grp[
                            "HISTORY_PASS"
                        ]
                    ).sum()
                ),

            "GAP_PASS_MEMBERS":
                int(
                    (
                        grp[
                            "IN_NIFTY_PHARMA_AT_FORMATION"
                        ]
                        &
                        grp[
                            "GAP_PASS"
                        ]
                    ).sum()
                ),

            "SEGMENT_PASS_MEMBERS":
                int(
                    (
                        grp[
                            "IN_NIFTY_PHARMA_AT_FORMATION"
                        ]
                        &
                        grp[
                            "SEGMENT_PASS"
                        ]
                    ).sum()
                ),

            "PRICE_PASS_MEMBERS":
                int(
                    (
                        grp[
                            "IN_NIFTY_PHARMA_AT_FORMATION"
                        ]
                        &
                        grp[
                            "PRICE_PASS"
                        ]
                    ).sum()
                ),

            "FORMATION_TRADABLE_MEMBERS":
                int(
                    (
                        grp[
                            "IN_NIFTY_PHARMA_AT_FORMATION"
                        ]
                        &
                        grp[
                            "FORMATION_TRADABILITY_PASS"
                        ]
                    ).sum()
                ),

            "PRE_LIQUIDITY_ELIGIBLE":
                int(
                    grp[
                        "PRE_LIQUIDITY_ELIGIBLE"
                    ].sum()
                ),
        }
    )


formation_counts = pd.DataFrame(
    count_records
)


# =============================================================================
# 11. EXCLUSION REASON COUNTS
# =============================================================================

exclusion_records = []


for formation_date, grp in member_candidates.groupby(
    "FORMATION_DATE"
):

    reasons = {
        "INSUFFICIENT_FORMATION_COVERAGE":
            int(
                (
                    ~grp[
                        "HISTORY_PASS"
                    ]
                ).sum()
            ),

        "EXCESSIVE_MISSING_SESSION_GAP":
            int(
                (
                    ~grp[
                        "GAP_PASS"
                    ]
                ).sum()
            ),

        "FORMATION_CROSSES_STRUCTURAL_SEGMENT":
            int(
                (
                    ~grp[
                        "SEGMENT_PASS"
                    ]
                ).sum()
            ),

        "MEDIAN_PRICE_BELOW_FLOOR":
            int(
                (
                    ~grp[
                        "PRICE_PASS"
                    ]
                ).sum()
            ),

        "NO_EQ_PRICE_ON_FORMATION_DATE":
            int(
                (
                    ~grp[
                        "FORMATION_TRADABILITY_PASS"
                    ]
                ).sum()
            ),
    }

    for reason, count in reasons.items():

        exclusion_records.append(
            {
                "FORMATION_DATE":
                    formation_date,

                "BLOCK_TYPE":
                    grp[
                        "BLOCK_TYPE"
                    ].iloc[0],

                "EXCLUSION_REASON":
                    reason,

                "MEMBER_STOCKS_FAILING":
                    count,
            }
        )


exclusion_counts = pd.DataFrame(
    exclusion_records
)


# =============================================================================
# 12. OOS BOUNDARY CHECK
# =============================================================================

oos_boundary = schedule[
    [
        "FORMATION_ANCHOR",
        "FORMATION_DATE",
        "TRADING_START",
        "TRADING_END",
        "BLOCK_TYPE",
    ]
].copy()


oos_boundary[
    "OOS_START"
] = OOS_START

oos_boundary[
    "OOS_END"
] = OOS_END


# =============================================================================
# 13. AUDIT SUMMARY
# =============================================================================

first_oos_rows = schedule[
    schedule[
        "BLOCK_TYPE"
    ].eq(
        "OOS"
    )
]


first_oos_formation = (
    first_oos_rows[
        "FORMATION_DATE"
    ].min()
    if not first_oos_rows.empty
    else pd.NaT
)


audit_summary = pd.DataFrame(
    [
        {
            "CHECK":
                "Point-in-time input rows",

            "VALUE":
                len(
                    data
                ),
        },
        {
            "CHECK":
                "Unique market sessions",

            "VALUE":
                data[
                    "DATE"
                ].nunique(),
        },
        {
            "CHECK":
                "Unique company identities",

            "VALUE":
                data[
                    "COMPANY_ID"
                ].nunique(),
        },
        {
            "CHECK":
                "Formation blocks",

            "VALUE":
                len(
                    schedule
                ),
        },
        {
            "CHECK":
                "Development blocks",

            "VALUE":
                int(
                    schedule[
                        "BLOCK_TYPE"
                    ].eq(
                        "DEVELOPMENT"
                    ).sum()
                ),
        },
        {
            "CHECK":
                "OOS blocks",

            "VALUE":
                int(
                    schedule[
                        "BLOCK_TYPE"
                    ].eq(
                        "OOS"
                    ).sum()
                ),
        },
        {
            "CHECK":
                "First OOS formation date",

            "VALUE":
                (
                    str(
                        first_oos_formation.date()
                    )
                    if pd.notna(
                        first_oos_formation
                    )
                    else ""
                ),
        },
        {
            "CHECK":
                "OOS trading start frozen",

            "VALUE":
                str(
                    OOS_START.date()
                ),
        },
        {
            "CHECK":
                "OOS trading end frozen",

            "VALUE":
                str(
                    OOS_END.date()
                ),
        },
        {
            "CHECK":
                "Coverage threshold pct",

            "VALUE":
                MIN_COVERAGE_PCT,
        },
        {
            "CHECK":
                "Maximum allowed missing-session gap",

            "VALUE":
                MAX_MISSING_GAP_SESSIONS,
        },
        {
            "CHECK":
                "Median close floor",

            "VALUE":
                MIN_MEDIAN_CLOSE,
        },
        {
            "CHECK":
                "Rows in stock-formation diagnostics",

            "VALUE":
                len(
                    diagnostics
                ),
        },
        {
            "CHECK":
                "Member candidate rows",

            "VALUE":
                len(
                    member_candidates
                ),
        },
        {
            "CHECK":
                "Pre-liquidity eligible rows",

            "VALUE":
                len(
                    pre_liquidity
                ),
        },
        {
            "CHECK":
                "Liquidity threshold chosen",

            "VALUE":
                "NO",
        },
        {
            "CHECK":
                "Historical F&O eligibility merged",

            "VALUE":
                "NO",
        },
        {
            "CHECK":
                "Final investable universe frozen",

            "VALUE":
                "NO",
        },
    ]
)


# =============================================================================
# 14. SAVE OUTPUTS
# =============================================================================

schedule.to_csv(
    OUTPUT_DIR
    /
    "01_FORMATION_DATE_SCHEDULE.csv",
    index=False,
    date_format="%Y-%m-%d"
)


diagnostics.to_csv(
    OUTPUT_DIR
    /
    "02_STOCK_FORMATION_DIAGNOSTICS.csv",
    index=False,
    date_format="%Y-%m-%d"
)


member_candidates.to_csv(
    OUTPUT_DIR
    /
    "03_MEMBER_CANDIDATES_ONLY.csv",
    index=False,
    date_format="%Y-%m-%d"
)


pre_liquidity.to_csv(
    OUTPUT_DIR
    /
    "04_PRE_LIQUIDITY_ELIGIBLE.csv",
    index=False,
    date_format="%Y-%m-%d"
)


liquidity_distribution.to_csv(
    OUTPUT_DIR
    /
    "05_DEVELOPMENT_LIQUIDITY_DISTRIBUTION.csv",
    index=False,
    date_format="%Y-%m-%d"
)


formation_counts.to_csv(
    OUTPUT_DIR
    /
    "06_FORMATION_DATE_COUNTS.csv",
    index=False,
    date_format="%Y-%m-%d"
)


exclusion_counts.to_csv(
    OUTPUT_DIR
    /
    "07_MEMBER_EXCLUSION_REASON_COUNTS.csv",
    index=False,
    date_format="%Y-%m-%d"
)


oos_boundary.to_csv(
    OUTPUT_DIR
    /
    "08_OOS_BOUNDARY_CHECK.csv",
    index=False,
    date_format="%Y-%m-%d"
)


audit_summary.to_csv(
    OUTPUT_DIR
    /
    "00_FORMATION_INVESTABILITY_AUDIT_SUMMARY.csv",
    index=False
)


# =============================================================================
# 15. README
# =============================================================================

readme = f"""
NIFTY PHARMA — FORMATION-DATE INVESTABILITY DIAGNOSTICS

INPUT
-----
{POINT_IN_TIME_FILE}

Historical membership:
{MEMBERSHIP_FILE}

FORMATION / TRADING DESIGN
--------------------------
Formation window:
{FORMATION_MONTHS} calendar months

Trading window:
{TRADING_MONTHS} calendar months

Formation anchors:
31 January and 31 July

Actual formation date:
Last observed NSE market session on or before the calendar anchor.

Formation window:
From calendar-anchor-minus-one-year-plus-one-day through actual formation date.

Trading start:
First observed NSE market session after the formation date.

Frozen OOS:
{OOS_START.date()} through {OOS_END.date()}

NON-LIQUIDITY RULES FROZEN AT THIS STAGE
----------------------------------------
Historical NIFTY Pharma member on formation date:
Required

Formation coverage:
At least {MIN_COVERAGE_PCT:.1f}%

Maximum consecutive missing market-session gap:
At most {MAX_MISSING_GAP_SESSIONS}

Structural continuity:
Exactly one SEGMENT_ID in the formation sample

Median raw CLOSE:
At least INR {MIN_MEDIAN_CLOSE:.2f}

Price on formation date:
Required

LIQUIDITY
---------
Liquidity is measured using raw NSE TOTAL_TRADED_VALUE.

This script calculates:
- median daily traded value
- 10th percentile daily traded value
- 25th percentile daily traded value
- average daily traded value

NO final rupee liquidity cutoff is chosen in this script.

The development-only liquidity distribution is written to:
05_DEVELOPMENT_LIQUIDITY_DISTRIBUTION.csv

Use DEVELOPMENT blocks only when choosing the fixed global liquidity threshold.
Do not use OOS results or pair-strategy performance to tune the threshold.

F&O / SHORTABILITY
------------------
Historical F&O eligibility is intentionally unresolved here.

Columns:
LIQUIDITY_PASS
FNO_ELIGIBLE
FINAL_INVESTABLE

remain missing until those rules are added.

IMPORTANT METHODOLOGY
---------------------
A stock does NOT need to have been a NIFTY Pharma constituent throughout the
whole formation window.

It only needs to be a member on the formation date. Earlier publicly available
price history may be used because it was observable at that time.

No price is forward-filled.

No pair selection or strategy-performance inspection occurs here.
"""

(
    OUTPUT_DIR
    /
    "README_FORMATION_INVESTABILITY.txt"
).write_text(
    readme,
    encoding="utf-8"
)


# =============================================================================
# 16. CONSOLE REPORT
# =============================================================================

print("\n")
print("=" * 110)
print("FORMATION-DATE INVESTABILITY DIAGNOSTICS COMPLETE")
print("=" * 110)

print("\nAUDIT SUMMARY")
print("-" * 110)

print(
    audit_summary.to_string(
        index=False
    )
)


print("\nFORMATION-DATE COUNTS")
print("-" * 110)

print(
    formation_counts.to_string(
        index=False
    )
)


print("\nOutputs saved to:")
print(
    OUTPUT_DIR
)


print("\nUpload these four files next:")
print(
    OUTPUT_DIR
    /
    "00_FORMATION_INVESTABILITY_AUDIT_SUMMARY.csv"
)

print(
    OUTPUT_DIR
    /
    "05_DEVELOPMENT_LIQUIDITY_DISTRIBUTION.csv"
)

print(
    OUTPUT_DIR
    /
    "06_FORMATION_DATE_COUNTS.csv"
)

print(
    OUTPUT_DIR
    /
    "07_MEMBER_EXCLUSION_REASON_COUNTS.csv"
)


print("\n")
print("=" * 110)
print(
    "LIQUIDITY THRESHOLD AND HISTORICAL F&O ELIGIBILITY ARE NOT YET FROZEN."
)
print("=" * 110)
