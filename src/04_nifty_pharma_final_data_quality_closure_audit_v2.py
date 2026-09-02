import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import hashlib
import warnings


# =============================================================================
# NIFTY PHARMA — FINAL DATA-QUALITY CLOSURE AUDIT (ROBUST VERSION)
# =============================================================================
#
# PURPOSE
# -------
# Final audit for the project's data-quality requirements:
#
# 1. ISIN / symbol continuity
# 2. Trading-calendar and special-session handling
# 3. Per-scrip missing-price / suspension-style gaps
# 4. Raw-field and row-key preservation
# 5. Reproducibility manifest
#
# IMPORTANT DESIGN CHANGE
# -----------------------
# This version DOES NOT abort merely because an audit item needs review.
#
# It only raises immediately for conditions that make the audit impossible,
# such as:
#   - required input file missing
#   - required columns missing
#   - invalid dates
#   - duplicate DATE + COMPANY_ID rows in a core dataset
#
# All other findings are written to CSV files and classified as:
#
#   PASS
#   INFO
#   REVIEW
#   FAIL
#
# Even if a FAIL exists, the program completes and saves the evidence so we
# can diagnose the exact issue rather than receiving only a final RuntimeError.
#
# NO DATA IS MODIFIED.
# NO FORWARD FILL IS PERFORMED.
# =============================================================================


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(r"C:\fin proj")

CLEAN_EQ_FILE = (
    PROJECT_ROOT
    / "nse_pharma_clean_base"
    / "NIFTY_PHARMA_EQ_BASE_2016_2026.csv"
)

TOTAL_RETURN_FILE = (
    PROJECT_ROOT
    / "nse_pharma_total_return"
    / "NIFTY_PHARMA_TOTAL_RETURN_BASE_2016_2026.csv"
)

POINT_IN_TIME_FILE = (
    PROJECT_ROOT
    / "nse_pharma_point_in_time_universe"
    / "NIFTY_PHARMA_POINT_IN_TIME_FLAGGED_2016_2026.csv"
)

MEMBERSHIP_FILENAME = "historical_nifty_pharma_FINAL_AUDITED.csv"

CORP_LEDGER_FILE = (
    PROJECT_ROOT
    / "nse_pharma_corporate_action_ledger"
    / "CORPORATE_ACTION_TREATMENT_LEDGER_2016_2026.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "nse_pharma_data_quality_closure_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. SETTINGS
# =============================================================================

AUDIT_START = pd.Timestamp("2016-01-01")
AUDIT_END   = pd.Timestamp("2026-07-31")

# SHA256 is useful for the final submission manifest.
# Set False if you only want a quick audit run.
COMPUTE_SHA256 = True

# Known Muhurat dates within our data period.
# These are documentary labels only. A missing one creates REVIEW, not a
# fatal error, because the audit should still finish and show the evidence.
KNOWN_MUHURAT_DATES = {
    pd.Timestamp("2016-10-30"),
    pd.Timestamp("2017-10-19"),
    pd.Timestamp("2018-11-07"),
    pd.Timestamp("2019-10-27"),
    pd.Timestamp("2020-11-14"),
    pd.Timestamp("2021-11-04"),
    pd.Timestamp("2022-10-24"),
    pd.Timestamp("2023-11-12"),
    pd.Timestamp("2024-11-01"),
    pd.Timestamp("2025-10-21"),
}

TRACKED_DIRS = [
    PROJECT_ROOT / "nse_pharma_2016_2026",
    PROJECT_ROOT / "nse_pharma_2016_2026_REPAIRED",
    PROJECT_ROOT / "nse_pharma_clean_base",
    PROJECT_ROOT / "nse_pharma_corporate_actions",
    PROJECT_ROOT / "nse_pharma_corporate_action_ledger",
    PROJECT_ROOT / "nse_pharma_total_return",
    PROJECT_ROOT / "nse_pharma_point_in_time_universe",
    PROJECT_ROOT / "nse_pharma_muhurat_repair",
]


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


def clean_str(series):
    return (
        series
        .astype("string")
        .str.strip()
    )


def clean_upper(series):
    return (
        clean_str(series)
        .str.upper()
    )


def normalize_membership_symbol(symbol):
    if pd.isna(symbol):
        return pd.NA

    s = str(symbol).strip().upper()

    # Known spelling mismatch in the audited membership file.
    if s == "AJANTAPHARM":
        return "AJANTPHARM"

    return s


def stable_company_id(symbol):
    if pd.isna(symbol):
        return pd.NA

    s = str(symbol).strip().upper()

    if s in {"CADILAHC", "ZYDUSLIFE"}:
        return "ZYDUS"

    if s == "AJANTAPHARM":
        return "AJANTPHARM"

    return s


def locate_membership_file():
    direct = PROJECT_ROOT / MEMBERSHIP_FILENAME

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
            "\nCould not find finalized membership file:\n"
            f"{MEMBERSHIP_FILENAME}\n"
            f"under {PROJECT_ROOT}"
        )

    raise RuntimeError(
        "\nMultiple copies of the finalized membership file were found.\n"
        "Choose one authoritative copy before continuing:\n\n"
        + "\n".join(str(x) for x in matches)
    )


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def safe_csv_row_count(path):
    if not path.exists():
        return np.nan

    if path.suffix.lower() != ".csv":
        return np.nan

    try:
        with open(path, "rb") as f:
            n = sum(1 for _ in f)

        return max(n - 1, 0)

    except Exception:
        return np.nan


def add_check(
    checks,
    category,
    check,
    value,
    status,
    interpretation,
    critical=False
):
    checks.append(
        {
            "CATEGORY": category,
            "CHECK": check,
            "VALUE": value,
            "STATUS": status,
            "CRITICAL": bool(critical),
            "INTERPRETATION": interpretation,
        }
    )


def detect_boolean(series):
    if series.dtype == bool:
        return series

    return (
        series
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


def build_gap_episodes(
    missing_rows,
    session_rank,
    group_cols
):
    """
    Consecutive means consecutive observed NSE market sessions,
    not consecutive calendar dates.
    """

    if missing_rows.empty:
        return pd.DataFrame()

    work = missing_rows.copy()

    work["_SESSION_NO"] = (
        work["DATE"]
        .map(session_rank)
    )

    work = work.sort_values(
        group_cols
        +
        ["_SESSION_NO"]
    )

    records = []

    for keys, grp in work.groupby(
        group_cols,
        dropna=False
    ):

        if not isinstance(keys, tuple):
            keys = (keys,)

        grp = grp.sort_values(
            "_SESSION_NO"
        ).copy()

        grp["_NEW_GAP"] = (
            grp["_SESSION_NO"]
            .diff()
            .ne(1)
        )

        grp["_GAP_ID"] = (
            grp["_NEW_GAP"]
            .cumsum()
        )

        for gap_no, ep in grp.groupby(
            "_GAP_ID"
        ):

            row = {
                col: key
                for col, key
                in zip(group_cols, keys)
            }

            row.update(
                {
                    "GAP_EPISODE":
                        int(gap_no),

                    "GAP_START_DATE":
                        ep["DATE"].min(),

                    "GAP_END_DATE":
                        ep["DATE"].max(),

                    "MISSING_MARKET_SESSIONS":
                        int(len(ep)),

                    "CALENDAR_DAYS_SPANNED":
                        int(
                            (
                                ep["DATE"].max()
                                -
                                ep["DATE"].min()
                            ).days
                            +
                            1
                        ),
                }
            )

            records.append(row)

    return pd.DataFrame(records)


def read_core_csv(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} file not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.replace(
            "\ufeff",
            "",
            regex=False
        )
        .str.strip()
        .str.upper()
    )

    return df


def comparable_numeric(a, b):
    a = pd.to_numeric(
        a,
        errors="coerce"
    )

    b = pd.to_numeric(
        b,
        errors="coerce"
    )

    both_missing = (
        a.isna()
        &
        b.isna()
    )

    both_present_equal = (
        a.notna()
        &
        b.notna()
        &
        np.isclose(
            a.fillna(0),
            b.fillna(0),
            rtol=0,
            atol=1e-9
        )
    )

    return (
        both_missing
        |
        both_present_equal
    )


# =============================================================================
# 4. LOCATE / LOAD INPUTS
# =============================================================================

MEMBERSHIP_FILE = locate_membership_file()

print("=" * 110)
print("NIFTY PHARMA FINAL DATA-QUALITY CLOSURE AUDIT — ROBUST VERSION")
print("=" * 110)

print("\nLoading datasets...")

clean_eq = read_core_csv(
    CLEAN_EQ_FILE,
    "Clean EQ"
)

total_return = read_core_csv(
    TOTAL_RETURN_FILE,
    "Total return"
)

pit = read_core_csv(
    POINT_IN_TIME_FILE,
    "Point-in-time"
)

membership = read_core_csv(
    MEMBERSHIP_FILE,
    "Historical membership"
)


# =============================================================================
# 5. HARD INPUT VALIDATION
# =============================================================================

core_required = {
    "DATE",
    "SYMBOL",
    "COMPANY_ID",
    "ISIN",
    "CLOSE",
}

for name, df in [
    ("CLEAN_EQ", clean_eq),
    ("TOTAL_RETURN", total_return),
    ("POINT_IN_TIME", pit),
]:

    missing = (
        core_required
        -
        set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{name} missing required columns:\n"
            f"{sorted(missing)}"
        )

    df["DATE"] = parse_dates(
        df["DATE"]
    )

    df["SYMBOL"] = clean_upper(
        df["SYMBOL"]
    )

    df["COMPANY_ID"] = clean_upper(
        df["COMPANY_ID"]
    )

    df["ISIN"] = clean_upper(
        df["ISIN"]
    )

    if df["DATE"].isna().any():
        raise ValueError(
            f"{name} contains invalid DATE values."
        )

    if df[
        ["DATE", "COMPANY_ID"]
    ].duplicated().any():

        bad = df[
            df[
                ["DATE", "COMPANY_ID"]
            ].duplicated(keep=False)
        ].copy()

        bad.to_csv(
            OUTPUT_DIR
            /
            f"FATAL_{name}_DUPLICATE_DATE_COMPANY.csv",
            index=False,
            date_format="%Y-%m-%d"
        )

        raise RuntimeError(
            f"{name} contains duplicate DATE + COMPANY_ID rows."
        )


membership_required = {
    "SYMBOL",
    "COMPANY",
    "START_DATE",
    "END_DATE",
}

missing_membership = (
    membership_required
    -
    set(membership.columns)
)

if missing_membership:
    raise ValueError(
        "Membership file missing required columns:\n"
        f"{sorted(missing_membership)}"
    )

membership["START_DATE"] = parse_dates(
    membership["START_DATE"]
)

membership["END_DATE"] = parse_dates(
    membership["END_DATE"]
)

if membership[
    ["START_DATE", "END_DATE"]
].isna().any().any():

    raise ValueError(
        "Membership file contains invalid START_DATE/END_DATE."
    )


# =============================================================================
# 6. AUDIT SUMMARY HOLDER
# =============================================================================

checks = []


# =============================================================================
# 7. ROW-COUNT / ROW-KEY PRESERVATION
# =============================================================================

print("Checking row/key preservation...")

clean_keys = set(
    zip(
        clean_eq["DATE"],
        clean_eq["COMPANY_ID"]
    )
)

tr_keys = set(
    zip(
        total_return["DATE"],
        total_return["COMPANY_ID"]
    )
)

pit_keys = set(
    zip(
        pit["DATE"],
        pit["COMPANY_ID"]
    )
)


row_key_audit = pd.DataFrame(
    [
        {
            "COMPARISON":
                "CLEAN_EQ -> TOTAL_RETURN",

            "LEFT_ROWS":
                len(clean_eq),

            "RIGHT_ROWS":
                len(total_return),

            "LEFT_ONLY_KEYS":
                len(clean_keys - tr_keys),

            "RIGHT_ONLY_KEYS":
                len(tr_keys - clean_keys),

            "EXACT_KEY_MATCH":
                clean_keys == tr_keys,
        },
        {
            "COMPARISON":
                "TOTAL_RETURN -> POINT_IN_TIME",

            "LEFT_ROWS":
                len(total_return),

            "RIGHT_ROWS":
                len(pit),

            "LEFT_ONLY_KEYS":
                len(tr_keys - pit_keys),

            "RIGHT_ONLY_KEYS":
                len(pit_keys - tr_keys),

            "EXACT_KEY_MATCH":
                tr_keys == pit_keys,
        },
    ]
)


key_match_pass = bool(
    row_key_audit[
        "EXACT_KEY_MATCH"
    ].all()
)

add_check(
    checks,
    "REPRODUCIBILITY",
    "DATE + COMPANY_ID keys preserved across transformation layers",
    int(key_match_pass),
    "PASS" if key_match_pass else "FAIL",
    (
        "PASS means later cleaning layers did not silently add "
        "forward-filled rows or drop existing observations."
    ),
    critical=True
)


# =============================================================================
# 8. RAW FIELD PRESERVATION
# =============================================================================

print("Checking raw NSE fields were not rewritten...")

candidate_raw_fields = [
    "SYMBOL",
    "ISIN",
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "LAST",
    "PREV_CLOSE",
    "TOTAL_TRADED_QTY",
    "TOTAL_TRADED_VALUE",
    "TOTAL_TRADES",
]

raw_fields = [
    c
    for c
    in candidate_raw_fields
    if (
        c in clean_eq.columns
        and
        c in pit.columns
    )
]


raw_join = clean_eq[
    ["DATE", "COMPANY_ID"]
    +
    raw_fields
].merge(
    pit[
        ["DATE", "COMPANY_ID"]
        +
        raw_fields
    ],
    on=[
        "DATE",
        "COMPANY_ID"
    ],
    how="outer",
    suffixes=(
        "_CLEAN",
        "_PIT"
    ),
    indicator=True,
    validate="one_to_one"
)


raw_field_records = []
raw_field_mismatches = []


for field in raw_fields:

    left = raw_join[
        field
        +
        "_CLEAN"
    ]

    right = raw_join[
        field
        +
        "_PIT"
    ]

    if field in {
        "SYMBOL",
        "ISIN"
    }:

        eq = (
            left.astype("string")
            .fillna("<NA>")
            .eq(
                right.astype("string")
                .fillna("<NA>")
            )
        )

    else:

        eq = comparable_numeric(
            left,
            right
        )

    mismatch = (
        ~eq
        |
        raw_join["_merge"].ne("both")
    )

    mismatch_count = int(
        mismatch.sum()
    )

    raw_field_records.append(
        {
            "FIELD":
                field,

            "ROWS_COMPARED":
                len(raw_join),

            "MISMATCH_ROWS":
                mismatch_count,

            "STATUS":
                (
                    "PASS"
                    if mismatch_count == 0
                    else "FAIL"
                ),
        }
    )

    if mismatch_count > 0:

        temp = raw_join.loc[
            mismatch,
            [
                "DATE",
                "COMPANY_ID",
                field
                +
                "_CLEAN",
                field
                +
                "_PIT",
                "_merge",
            ]
        ].copy()

        temp[
            "FIELD"
        ] = field

        raw_field_mismatches.append(
            temp
        )


raw_field_audit = pd.DataFrame(
    raw_field_records
)


raw_fields_pass = (
    raw_field_audit[
        "MISMATCH_ROWS"
    ].sum() == 0
)


add_check(
    checks,
    "REPRODUCIBILITY",
    "Raw NSE market fields preserved from clean EQ to point-in-time layer",
    int(
        raw_field_audit[
            "MISMATCH_ROWS"
        ].sum()
    ),
    "PASS" if raw_fields_pass else "FAIL",
    (
        "Any mismatch means a raw field such as CLOSE, volume, "
        "traded value, ISIN or OHLC changed downstream."
    ),
    critical=True
)


# =============================================================================
# 9. ISIN CONTINUITY
# =============================================================================

print("Auditing ISIN continuity...")

missing_isin = pit[
    pit["ISIN"].isna()
    |
    pit["ISIN"].eq("")
].copy()


company_isin_summary = (
    pit[
        pit["ISIN"].notna()
        &
        ~pit["ISIN"].eq("")
    ]
    .groupby(
        "COMPANY_ID",
        as_index=False
    )
    .agg(
        UNIQUE_ISINS=(
            "ISIN",
            "nunique"
        ),

        UNIQUE_SYMBOLS=(
            "SYMBOL",
            "nunique"
        ),

        FIRST_DATE=(
            "DATE",
            "min"
        ),

        LAST_DATE=(
            "DATE",
            "max"
        ),

        ISINS=(
            "ISIN",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            x.dropna()
                        )
                    )
                )
        ),

        SYMBOLS=(
            "SYMBOL",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            x.dropna()
                        )
                    )
                )
        ),
    )
)


symbol_identity_summary = (
    pit
    .groupby(
        "SYMBOL",
        as_index=False
    )
    .agg(
        UNIQUE_COMPANY_IDS=(
            "COMPANY_ID",
            "nunique"
        ),

        UNIQUE_ISINS=(
            "ISIN",
            "nunique"
        ),

        FIRST_DATE=(
            "DATE",
            "min"
        ),

        LAST_DATE=(
            "DATE",
            "max"
        ),

        COMPANY_IDS=(
            "COMPANY_ID",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            x.dropna()
                        )
                    )
                )
        ),

        ISINS=(
            "ISIN",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            x.dropna()
                        )
                    )
                )
        ),
    )
)


isin_identity_summary = (
    pit[
        pit["ISIN"].notna()
        &
        ~pit["ISIN"].eq("")
    ]
    .groupby(
        "ISIN",
        as_index=False
    )
    .agg(
        UNIQUE_COMPANY_IDS=(
            "COMPANY_ID",
            "nunique"
        ),

        UNIQUE_SYMBOLS=(
            "SYMBOL",
            "nunique"
        ),

        COMPANY_IDS=(
            "COMPANY_ID",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            x.dropna()
                        )
                    )
                )
        ),

        SYMBOLS=(
            "SYMBOL",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            x.dropna()
                        )
                    )
                )
        ),

        FIRST_DATE=(
            "DATE",
            "min"
        ),

        LAST_DATE=(
            "DATE",
            "max"
        ),
    )
)


symbol_reuse_review = symbol_identity_summary[
    symbol_identity_summary[
        "UNIQUE_COMPANY_IDS"
    ] > 1
].copy()


isin_cross_company_review = isin_identity_summary[
    isin_identity_summary[
        "UNIQUE_COMPANY_IDS"
    ] > 1
].copy()


# Detect chronological ISIN transitions within a stable company.
transition_records = []

for company_id, grp in pit.groupby(
    "COMPANY_ID"
):

    g = (
        grp[
            [
                "DATE",
                "SYMBOL",
                "ISIN",
            ]
        ]
        .sort_values(
            "DATE"
        )
        .reset_index(
            drop=True
        )
    )

    prev_isin = None
    prev_symbol = None
    prev_date = None

    for _, row in g.iterrows():

        current_isin = row["ISIN"]

        if (
            pd.isna(current_isin)
            or
            str(current_isin).strip() == ""
        ):
            continue

        if prev_isin is None:

            prev_isin = current_isin
            prev_symbol = row["SYMBOL"]
            prev_date = row["DATE"]
            continue

        if current_isin != prev_isin:

            transition_records.append(
                {
                    "COMPANY_ID":
                        company_id,

                    "OLD_SYMBOL":
                        prev_symbol,

                    "NEW_SYMBOL":
                        row["SYMBOL"],

                    "OLD_ISIN":
                        prev_isin,

                    "NEW_ISIN":
                        current_isin,

                    "LAST_DATE_OLD_ISIN":
                        prev_date,

                    "FIRST_DATE_NEW_ISIN":
                        row["DATE"],
                }
            )

            prev_isin = current_isin

        prev_symbol = row["SYMBOL"]
        prev_date = row["DATE"]


isin_transitions = pd.DataFrame(
    transition_records
)


if isin_transitions.empty:

    isin_transitions = pd.DataFrame(
        columns=[
            "COMPANY_ID",
            "OLD_SYMBOL",
            "NEW_SYMBOL",
            "OLD_ISIN",
            "NEW_ISIN",
            "LAST_DATE_OLD_ISIN",
            "FIRST_DATE_NEW_ISIN",
        ]
    )


# Corporate-action proximity annotation.
if CORP_LEDGER_FILE.exists():

    corp = read_core_csv(
        CORP_LEDGER_FILE,
        "Corporate-action ledger"
    )

    if {
        "EX_DATE",
        "COMPANY_ID"
    }.issubset(
        corp.columns
    ):

        corp["EX_DATE"] = parse_dates(
            corp["EX_DATE"]
        )

        corp["COMPANY_ID"] = clean_upper(
            corp["COMPANY_ID"]
        )

        annotations = []

        for _, row in isin_transitions.iterrows():

            sub = corp[
                corp["COMPANY_ID"].eq(
                    row["COMPANY_ID"]
                )
                &
                corp["EX_DATE"].notna()
            ].copy()

            nearest_date = pd.NaT
            nearest_days = np.nan
            nearest_class = ""
            nearest_purpose = ""

            if not sub.empty:

                sub["_DIST"] = (
                    sub["EX_DATE"]
                    -
                    row[
                        "FIRST_DATE_NEW_ISIN"
                    ]
                ).abs().dt.days

                best = sub.sort_values(
                    "_DIST"
                ).iloc[0]

                nearest_days = int(
                    best["_DIST"]
                )

                if nearest_days <= 15:

                    nearest_date = best[
                        "EX_DATE"
                    ]

                    nearest_class = str(
                        best.get(
                            "PRIMARY_CLASS",
                            ""
                        )
                    )

                    nearest_purpose = str(
                        best.get(
                            "PURPOSE",
                            ""
                        )
                    )

            annotations.append(
                {
                    "NEAREST_CORP_ACTION_DATE":
                        nearest_date,

                    "CORP_ACTION_DISTANCE_DAYS":
                        nearest_days,

                    "NEAREST_CORP_ACTION_CLASS":
                        nearest_class,

                    "NEAREST_CORP_ACTION_PURPOSE":
                        nearest_purpose,

                    "REVIEW_CLASS":
                        (
                            "LIKELY_EXPLAINED_BY_NEARBY_CORPORATE_ACTION"
                            if pd.notna(nearest_date)
                            else "MANUAL_REVIEW"
                        ),
                }
            )

        if annotations:

            isin_transitions = pd.concat(
                [
                    isin_transitions.reset_index(drop=True),
                    pd.DataFrame(annotations),
                ],
                axis=1
            )


if "REVIEW_CLASS" not in isin_transitions.columns:

    isin_transitions["REVIEW_CLASS"] = (
        "MANUAL_REVIEW"
        if len(isin_transitions) > 0
        else pd.Series(
            dtype="string"
        )
    )


unexplained_isin_transitions = isin_transitions[
    isin_transitions[
        "REVIEW_CLASS"
    ].eq(
        "MANUAL_REVIEW"
    )
].copy()


add_check(
    checks,
    "SYMBOL_ISIN",
    "Rows with missing ISIN",
    len(missing_isin),
    (
        "PASS"
        if len(missing_isin) == 0
        else "REVIEW"
    ),
    (
        "Missing ISINs do not automatically invalidate the dataset, "
        "but they must be visible in the audit."
    )
)


add_check(
    checks,
    "SYMBOL_ISIN",
    "Symbols mapping to multiple stable COMPANY_IDs",
    len(symbol_reuse_review),
    (
        "PASS"
        if len(symbol_reuse_review) == 0
        else "REVIEW"
    ),
    (
        "Potential ticker reuse. REVIEW rather than automatic failure "
        "because corporate restructurings can create legitimate cases."
    )
)


add_check(
    checks,
    "SYMBOL_ISIN",
    "ISINs mapping to multiple stable COMPANY_IDs",
    len(isin_cross_company_review),
    (
        "PASS"
        if len(isin_cross_company_review) == 0
        else "REVIEW"
    ),
    (
        "Potential identity-map issue. This is surfaced for inspection "
        "rather than causing the entire audit to abort."
    )
)


add_check(
    checks,
    "SYMBOL_ISIN",
    "Stable companies with more than one historical ISIN",
    int(
        (
            company_isin_summary[
                "UNIQUE_ISINS"
            ] > 1
        ).sum()
    ),
    "INFO",
    (
        "Multiple ISINs can be legitimate after splits, consolidations "
        "or restructurings. Inspect the transition file."
    )
)


add_check(
    checks,
    "SYMBOL_ISIN",
    "Unexplained chronological ISIN transitions",
    len(unexplained_isin_transitions),
    (
        "PASS"
        if len(unexplained_isin_transitions) == 0
        else "REVIEW"
    ),
    (
        "These require a quick manual explanation before final memo wording."
    )
)


# =============================================================================
# 10. MASTER TRADING CALENDAR
# =============================================================================

print("Building master trading-calendar audit...")

clean_dates = set(
    clean_eq["DATE"].unique()
)

tr_dates = set(
    total_return["DATE"].unique()
)

pit_dates = set(
    pit["DATE"].unique()
)


calendar_date_set_consistent = (
    clean_dates == tr_dates == pit_dates
)


add_check(
    checks,
    "TRADING_CALENDAR",
    "Observed market-session dates identical across layers",
    int(
        calendar_date_set_consistent
    ),
    (
        "PASS"
        if calendar_date_set_consistent
        else "FAIL"
    ),
    (
        "Later layers should not create or remove entire market-session dates."
    ),
    critical=True
)


observed_market_dates = pd.Series(
    sorted(clean_dates),
    name="DATE"
)


calendar = pd.DataFrame(
    {
        "DATE":
            pd.date_range(
                AUDIT_START,
                AUDIT_END,
                freq="D"
            )
    }
)


calendar["DAY_NAME"] = (
    calendar["DATE"]
    .dt.day_name()
)

calendar["IS_WEEKEND"] = (
    calendar["DATE"]
    .dt.dayofweek >= 5
)

calendar["IS_MARKET_SESSION"] = (
    calendar["DATE"]
    .isin(clean_dates)
)

calendar["IS_KNOWN_MUHURAT_DATE"] = (
    calendar["DATE"]
    .isin(KNOWN_MUHURAT_DATES)
)


def session_class(row):

    if row["IS_MARKET_SESSION"]:

        if row[
            "IS_KNOWN_MUHURAT_DATE"
        ]:
            return "MUHURAT_TRADING_SESSION"

        if row["IS_WEEKEND"]:
            return "OTHER_SPECIAL_WEEKEND_TRADING_SESSION"

        return "WEEKDAY_TRADING_SESSION"

    if row["IS_WEEKEND"]:
        return "WEEKEND_NO_SESSION"

    return "WEEKDAY_NO_SESSION"


calendar["SESSION_CLASS"] = (
    calendar.apply(
        session_class,
        axis=1
    )
)


muhurat_audit = calendar[
    calendar["IS_KNOWN_MUHURAT_DATE"]
].copy()


# A known Muhurat date after AUDIT_END should not be in calendar anyway;
# this condition is retained for clarity.
muhurat_audit["WITHIN_AUDIT_WINDOW"] = (
    muhurat_audit["DATE"].between(
        AUDIT_START,
        AUDIT_END,
        inclusive="both"
    )
)


muhurat_missing = muhurat_audit[
    muhurat_audit["WITHIN_AUDIT_WINDOW"]
    &
    ~muhurat_audit["IS_MARKET_SESSION"]
].copy()


weekend_special_sessions = calendar[
    calendar["IS_MARKET_SESSION"]
    &
    calendar["IS_WEEKEND"]
].copy()


weekday_no_session = calendar[
    ~calendar["IS_MARKET_SESSION"]
    &
    ~calendar["IS_WEEKEND"]
].copy()


add_check(
    checks,
    "TRADING_CALENDAR",
    "Observed market sessions",
    len(observed_market_dates),
    "INFO",
    (
        "Derived directly from cleaned NSE EQ observations; no external "
        "holiday calendar is used to fabricate sessions."
    )
)


add_check(
    checks,
    "TRADING_CALENDAR",
    "Known Muhurat dates absent from observed NSE sessions",
    len(muhurat_missing),
    (
        "PASS"
        if len(muhurat_missing) == 0
        else "REVIEW"
    ),
    (
        "A missing known special session is surfaced for investigation, "
        "but no synthetic price/session is created."
    )
)


add_check(
    checks,
    "TRADING_CALENDAR",
    "Observed weekend trading sessions",
    len(weekend_special_sessions),
    "INFO",
    (
        "Includes Muhurat and any other actual special weekend sessions."
    )
)


add_check(
    checks,
    "TRADING_CALENDAR",
    "Weekday dates with no observed exchange session",
    len(weekday_no_session),
    "INFO",
    (
        "These are treated as exchange-wide non-session dates; "
        "the audit does not need to assign a holiday name to each."
    )
)


# =============================================================================
# 11. PER-SCRIP GAP AUDIT WHILE HISTORICALLY A MEMBER
# =============================================================================

print("Auditing per-scrip missing observations...")

membership["MEMBERSHIP_SYMBOL_SOURCE"] = clean_upper(
    membership["SYMBOL"]
)

membership["MEMBERSHIP_SYMBOL"] = (
    membership[
        "MEMBERSHIP_SYMBOL_SOURCE"
    ]
    .map(
        normalize_membership_symbol
    )
    .astype("string")
)

membership["COMPANY_ID"] = (
    membership[
        "MEMBERSHIP_SYMBOL"
    ]
    .map(
        stable_company_id
    )
    .astype("string")
)

membership["MEMBERSHIP_PERIOD_ID"] = [
    f"NM{i:04d}"
    for i in range(
        1,
        len(membership) + 1
    )
]


market_dates = pd.Series(
    sorted(clean_dates),
    name="DATE"
)


member_expected_parts = []


for _, row in membership.iterrows():

    dates = market_dates[
        market_dates.between(
            row["START_DATE"],
            row["END_DATE"],
            inclusive="both"
        )
    ]

    if dates.empty:
        continue

    temp = pd.DataFrame(
        {
            "DATE":
                dates.values
        }
    )

    temp["COMPANY_ID"] = (
        row["COMPANY_ID"]
    )

    temp["MEMBERSHIP_PERIOD_ID"] = (
        row["MEMBERSHIP_PERIOD_ID"]
    )

    temp["MEMBERSHIP_SYMBOL"] = (
        row["MEMBERSHIP_SYMBOL"]
    )

    temp["MEMBERSHIP_START_DATE"] = (
        row["START_DATE"]
    )

    temp["MEMBERSHIP_END_DATE"] = (
        row["END_DATE"]
    )

    member_expected_parts.append(
        temp
    )


member_expected = pd.concat(
    member_expected_parts,
    ignore_index=True
)


if member_expected[
    ["DATE", "COMPANY_ID"]
].duplicated().any():

    duplicate_expected = member_expected[
        member_expected[
            ["DATE", "COMPANY_ID"]
        ].duplicated(keep=False)
    ].copy()

    duplicate_expected.to_csv(
        OUTPUT_DIR
        /
        "REVIEW_DUPLICATE_EXPECTED_MEMBERSHIP_ROWS.csv",
        index=False,
        date_format="%Y-%m-%d"
    )

    warnings.warn(
        "Historical membership intervals overlap after stable-company "
        "mapping. See REVIEW_DUPLICATE_EXPECTED_MEMBERSHIP_ROWS.csv"
    )


actual_prices = pit[
    [
        "DATE",
        "COMPANY_ID",
        "SYMBOL",
        "CLOSE",
    ]
].copy()


member_expected = member_expected.merge(
    actual_prices,
    on=[
        "DATE",
        "COMPANY_ID"
    ],
    how="left",
    validate=(
        "many_to_one"
        if member_expected[
            ["DATE", "COMPANY_ID"]
        ].duplicated().any()
        else "one_to_one"
    )
)


member_expected["PRICE_ROW_PRESENT"] = (
    member_expected["CLOSE"].notna()
)


member_missing = member_expected[
    ~member_expected[
        "PRICE_ROW_PRESENT"
    ]
].copy()


session_rank = {
    date: i
    for i, date
    in enumerate(
        sorted(clean_dates),
        start=1
    )
}


member_gap_episodes = build_gap_episodes(
    member_missing,
    session_rank,
    [
        "COMPANY_ID",
        "MEMBERSHIP_PERIOD_ID",
    ]
)


# Tag gaps near structural breaks.
if (
    not member_gap_episodes.empty
    and
    "STRUCTURAL_BREAK_FLAG"
    in total_return.columns
):

    sb_mask = detect_boolean(
        total_return[
            "STRUCTURAL_BREAK_FLAG"
        ]
    )

    sb = total_return.loc[
        sb_mask,
        [
            "COMPANY_ID",
            "DATE",
        ]
    ].rename(
        columns={
            "DATE":
                "STRUCTURAL_BREAK_ROW_DATE"
        }
    )

    contexts = []

    for _, gap in member_gap_episodes.iterrows():

        sub = sb[
            sb["COMPANY_ID"].eq(
                gap["COMPANY_ID"]
            )
        ].copy()

        nearest = pd.NaT
        distance = np.nan

        if not sub.empty:

            sub["_DIST"] = np.minimum(
                (
                    sub[
                        "STRUCTURAL_BREAK_ROW_DATE"
                    ]
                    -
                    gap["GAP_START_DATE"]
                ).abs().dt.days,

                (
                    sub[
                        "STRUCTURAL_BREAK_ROW_DATE"
                    ]
                    -
                    gap["GAP_END_DATE"]
                ).abs().dt.days
            )

            best = sub.sort_values(
                "_DIST"
            ).iloc[0]

            nearest = best[
                "STRUCTURAL_BREAK_ROW_DATE"
            ]

            distance = int(
                best["_DIST"]
            )

        contexts.append(
            {
                "NEAREST_STRUCTURAL_BREAK_DATE":
                    nearest,

                "STRUCTURAL_BREAK_DISTANCE_DAYS":
                    distance,

                "GAP_CONTEXT":
                    (
                        "NEAR_STRUCTURAL_BREAK"
                        if (
                            pd.notna(distance)
                            and
                            distance <= 20
                        )
                        else "PER_SCRIP_MISSING_PRICE_REVIEW"
                    ),
            }
        )

    member_gap_episodes = pd.concat(
        [
            member_gap_episodes.reset_index(drop=True),
            pd.DataFrame(contexts),
        ],
        axis=1
    )


add_check(
    checks,
    "TRADING_CALENDAR",
    "Missing member-date price rows",
    len(member_missing),
    "INFO",
    (
        "Expected member-date observations with no EQ price. "
        "They are deliberately left missing, never forward-filled."
    )
)


add_check(
    checks,
    "TRADING_CALENDAR",
    "Member per-scrip gap episodes",
    len(member_gap_episodes),
    "INFO",
    (
        "Consecutive missing observations measured in actual market sessions."
    )
)


add_check(
    checks,
    "TRADING_CALENDAR",
    "Maximum per-scrip member gap length in market sessions",
    (
        int(
            member_gap_episodes[
                "MISSING_MARKET_SESSIONS"
            ].max()
        )
        if not member_gap_episodes.empty
        else 0
    ),
    "INFO",
    (
        "Useful later for the formation-window investability filter."
    )
)


# =============================================================================
# 12. REPRODUCIBILITY MANIFEST
# =============================================================================

print("Building reproducibility file manifest...")

manifest_records = []
seen = set()


for root in TRACKED_DIRS:

    if not root.exists():

        manifest_records.append(
            {
                "TRACKED_ROOT":
                    str(root),

                "RELATIVE_PATH":
                    "",

                "ABSOLUTE_PATH":
                    str(root),

                "EXISTS":
                    False,

                "FILE_SIZE_BYTES":
                    np.nan,

                "MODIFIED_LOCAL":
                    "",

                "SHA256":
                    "",

                "HASH_STATUS":
                    "ROOT_MISSING",
            }
        )

        continue

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        resolved = str(
            path.resolve()
        ).lower()

        if resolved in seen:
            continue

        seen.add(resolved)

        stat = path.stat()

        sha = ""
        hash_status = "NOT_REQUESTED"

        if COMPUTE_SHA256:

            try:
                sha = sha256_file(path)
                hash_status = "OK"

            except Exception as exc:
                hash_status = (
                    "ERROR: "
                    +
                    str(exc)[:180]
                )

        manifest_records.append(
            {
                "TRACKED_ROOT":
                    str(root),

                "RELATIVE_PATH":
                    str(
                        path.relative_to(root)
                    ),

                "ABSOLUTE_PATH":
                    str(path),

                "EXISTS":
                    True,

                "FILE_SIZE_BYTES":
                    int(stat.st_size),

                "MODIFIED_LOCAL":
                    datetime.fromtimestamp(
                        stat.st_mtime
                    ).isoformat(
                        timespec="seconds"
                    ),

                "SHA256":
                    sha,

                "HASH_STATUS":
                    hash_status,
            }
        )


file_manifest = pd.DataFrame(
    manifest_records
)


key_files = [
    ("CLEAN_EQ_BASE", CLEAN_EQ_FILE),
    ("TOTAL_RETURN", TOTAL_RETURN_FILE),
    ("POINT_IN_TIME", POINT_IN_TIME_FILE),
    ("HISTORICAL_MEMBERSHIP", MEMBERSHIP_FILE),
    ("CORPORATE_ACTION_LEDGER", CORP_LEDGER_FILE),
]


key_manifest_records = []


for stage, path in key_files:

    exists = path.exists()

    sha = ""
    hash_status = (
        "MISSING"
        if not exists
        else "NOT_REQUESTED"
    )

    if exists and COMPUTE_SHA256:

        try:
            sha = sha256_file(path)
            hash_status = "OK"

        except Exception as exc:
            hash_status = (
                "ERROR: "
                +
                str(exc)[:180]
            )

    key_manifest_records.append(
        {
            "STAGE":
                stage,

            "PATH":
                str(path),

            "EXISTS":
                exists,

            "CSV_ROW_COUNT":
                (
                    safe_csv_row_count(path)
                    if exists
                    else np.nan
                ),

            "FILE_SIZE_BYTES":
                (
                    path.stat().st_size
                    if exists
                    else np.nan
                ),

            "SHA256":
                sha,

            "HASH_STATUS":
                hash_status,
        }
    )


key_dataset_manifest = pd.DataFrame(
    key_manifest_records
)


# Inventory saved Python code.
script_records = []


for path in PROJECT_ROOT.rglob("*.py"):

    if not path.is_file():
        continue

    stat = path.stat()

    sha = ""

    if COMPUTE_SHA256:

        try:
            sha = sha256_file(path)

        except Exception:
            sha = "HASH_ERROR"

    script_records.append(
        {
            "PATH":
                str(path),

            "RELATIVE_TO_PROJECT":
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "FILE_SIZE_BYTES":
                int(stat.st_size),

            "MODIFIED_LOCAL":
                datetime.fromtimestamp(
                    stat.st_mtime
                ).isoformat(
                    timespec="seconds"
                ),

            "SHA256":
                sha,
        }
    )


python_script_manifest = pd.DataFrame(
    script_records
)


if python_script_manifest.empty:

    python_script_manifest = pd.DataFrame(
        columns=[
            "PATH",
            "RELATIVE_TO_PROJECT",
            "FILE_SIZE_BYTES",
            "MODIFIED_LOCAL",
            "SHA256",
        ]
    )


hash_errors = 0

if not file_manifest.empty:
    hash_errors = int(
        file_manifest[
            "HASH_STATUS"
        ]
        .astype(str)
        .str.startswith(
            "ERROR"
        )
        .sum()
    )


add_check(
    checks,
    "REPRODUCIBILITY",
    "Key datasets present",
    int(
        key_dataset_manifest[
            "EXISTS"
        ].sum()
    ),
    (
        "PASS"
        if key_dataset_manifest[
            "EXISTS"
        ].all()
        else "REVIEW"
    ),
    (
        "All major raw-to-research stages should be identifiable in the manifest."
    )
)


add_check(
    checks,
    "REPRODUCIBILITY",
    "Tracked file-manifest rows",
    len(file_manifest),
    "INFO",
    (
        "Recursive inventory of raw, repaired, clean and derived data folders."
    )
)


add_check(
    checks,
    "REPRODUCIBILITY",
    "SHA256 hash errors",
    hash_errors,
    (
        "PASS"
        if hash_errors == 0
        else "REVIEW"
    ),
    (
        "Hash errors do not change data; they mean a file could not be "
        "fingerprinted and should be checked."
    )
)


add_check(
    checks,
    "REPRODUCIBILITY",
    "Python scripts saved under project root",
    len(python_script_manifest),
    (
        "PASS"
        if len(python_script_manifest) > 0
        else "REVIEW"
    ),
    (
        "If zero, save the pipeline scripts into C:\\fin proj before submission."
    )
)


# =============================================================================
# 13. FINAL CLOSURE SUMMARY
# =============================================================================

closure_summary = pd.DataFrame(
    checks
)


critical_failures = closure_summary[
    closure_summary["CRITICAL"]
    &
    closure_summary["STATUS"].eq("FAIL")
].copy()


review_items = closure_summary[
    closure_summary["STATUS"].eq("REVIEW")
].copy()


if not critical_failures.empty:

    overall_status = (
        "FAIL_REVIEW_REQUIRED"
    )

elif not review_items.empty:

    overall_status = (
        "PASS_WITH_REVIEW_ITEMS"
    )

else:

    overall_status = "PASS"


# =============================================================================
# 14. SAVE OUTPUTS
# =============================================================================

closure_summary.to_csv(
    OUTPUT_DIR
    /
    "00_DATA_QUALITY_CLOSURE_SUMMARY.csv",
    index=False
)


row_key_audit.to_csv(
    OUTPUT_DIR
    /
    "01_ROW_KEY_PRESERVATION_AUDIT.csv",
    index=False
)


raw_field_audit.to_csv(
    OUTPUT_DIR
    /
    "02_RAW_FIELD_PRESERVATION_AUDIT.csv",
    index=False
)


if raw_field_mismatches:

    pd.concat(
        raw_field_mismatches,
        ignore_index=True
    ).to_csv(
        OUTPUT_DIR
        /
        "03_RAW_FIELD_MISMATCH_DETAILS.csv",
        index=False,
        date_format="%Y-%m-%d"
    )

else:

    pd.DataFrame(
        columns=[
            "DATE",
            "COMPANY_ID",
            "FIELD",
        ]
    ).to_csv(
        OUTPUT_DIR
        /
        "03_RAW_FIELD_MISMATCH_DETAILS.csv",
        index=False
    )


company_isin_summary.to_csv(
    OUTPUT_DIR
    /
    "04_COMPANY_ISIN_SUMMARY.csv",
    index=False,
    date_format="%Y-%m-%d"
)


symbol_identity_summary.to_csv(
    OUTPUT_DIR
    /
    "05_SYMBOL_IDENTITY_SUMMARY.csv",
    index=False,
    date_format="%Y-%m-%d"
)


isin_identity_summary.to_csv(
    OUTPUT_DIR
    /
    "06_ISIN_IDENTITY_SUMMARY.csv",
    index=False,
    date_format="%Y-%m-%d"
)


isin_transitions.to_csv(
    OUTPUT_DIR
    /
    "07_ISIN_TRANSITIONS.csv",
    index=False,
    date_format="%Y-%m-%d"
)


unexplained_isin_transitions.to_csv(
    OUTPUT_DIR
    /
    "08_UNEXPLAINED_ISIN_TRANSITIONS.csv",
    index=False,
    date_format="%Y-%m-%d"
)


missing_isin.to_csv(
    OUTPUT_DIR
    /
    "09_MISSING_ISIN_ROWS.csv",
    index=False,
    date_format="%Y-%m-%d"
)


symbol_reuse_review.to_csv(
    OUTPUT_DIR
    /
    "10_SYMBOL_REUSE_REVIEW.csv",
    index=False,
    date_format="%Y-%m-%d"
)


isin_cross_company_review.to_csv(
    OUTPUT_DIR
    /
    "11_ISIN_CROSS_COMPANY_REVIEW.csv",
    index=False,
    date_format="%Y-%m-%d"
)


calendar.to_csv(
    OUTPUT_DIR
    /
    "12_MASTER_TRADING_CALENDAR.csv",
    index=False,
    date_format="%Y-%m-%d"
)


muhurat_audit.to_csv(
    OUTPUT_DIR
    /
    "13_MUHURAT_SESSION_AUDIT.csv",
    index=False,
    date_format="%Y-%m-%d"
)


weekend_special_sessions.to_csv(
    OUTPUT_DIR
    /
    "14_WEEKEND_SPECIAL_SESSIONS.csv",
    index=False,
    date_format="%Y-%m-%d"
)


weekday_no_session.to_csv(
    OUTPUT_DIR
    /
    "15_WEEKDAY_NO_SESSION_DATES.csv",
    index=False,
    date_format="%Y-%m-%d"
)


member_missing.to_csv(
    OUTPUT_DIR
    /
    "16_MEMBER_MISSING_MARKET_SESSIONS.csv",
    index=False,
    date_format="%Y-%m-%d"
)


member_gap_episodes.to_csv(
    OUTPUT_DIR
    /
    "17_MEMBER_GAP_EPISODES.csv",
    index=False,
    date_format="%Y-%m-%d"
)


file_manifest.to_csv(
    OUTPUT_DIR
    /
    "18_REPRODUCIBILITY_FILE_MANIFEST.csv",
    index=False
)


key_dataset_manifest.to_csv(
    OUTPUT_DIR
    /
    "19_KEY_DATASET_MANIFEST.csv",
    index=False
)


python_script_manifest.to_csv(
    OUTPUT_DIR
    /
    "20_PYTHON_SCRIPT_MANIFEST.csv",
    index=False
)


critical_failures.to_csv(
    OUTPUT_DIR
    /
    "21_CRITICAL_FAILURES.csv",
    index=False
)


review_items.to_csv(
    OUTPUT_DIR
    /
    "22_REVIEW_ITEMS.csv",
    index=False
)


status_file = pd.DataFrame(
    [
        {
            "OVERALL_STATUS":
                overall_status,

            "CRITICAL_FAILURE_COUNT":
                len(
                    critical_failures
                ),

            "REVIEW_ITEM_COUNT":
                len(
                    review_items
                ),

            "AUDIT_COMPLETED":
                True,

            "DATA_MODIFIED":
                False,

            "FORWARD_FILL_USED":
                False,
        }
    ]
)


status_file.to_csv(
    OUTPUT_DIR
    /
    "23_OVERALL_STATUS.csv",
    index=False
)


# =============================================================================
# 15. README
# =============================================================================

readme = f"""
NIFTY PHARMA FINAL DATA-QUALITY CLOSURE AUDIT — V2

OVERALL STATUS
--------------
{overall_status}

WHY THIS VERSION IS SAFER
-------------------------
The earlier audit stopped with a RuntimeError whenever a check classified as
critical failed. That made diagnosis unnecessarily difficult.

This version distinguishes:

PASS
    Requirement is satisfied by the evidence.

INFO
    Descriptive audit result; not an error.

REVIEW
    An observation needs explanation, but the program still finishes.

FAIL
    A serious inconsistency was detected.

The program DOES NOT throw a final RuntimeError simply because a FAIL or REVIEW
exists. It saves the exact evidence in CSV files first.

It still raises immediately for genuinely unusable inputs such as missing
required files, missing required columns, invalid dates, or duplicate
DATE + COMPANY_ID rows.

DATA QUALITY REQUIREMENTS COVERED
---------------------------------
1. Corporate-action adjusted pipeline preservation
2. Total-return layer preservation
3. Symbol / ISIN continuity
4. Trading calendar and special sessions
5. Per-scrip missing observations / suspensions
6. No-forward-fill evidence
7. Reproducibility manifest and hashes

NO DATA IS MODIFIED BY THIS PROGRAM.

OUTPUT FOLDER
-------------
{OUTPUT_DIR}

MOST IMPORTANT FILES
--------------------
00_DATA_QUALITY_CLOSURE_SUMMARY.csv
07_ISIN_TRANSITIONS.csv
08_UNEXPLAINED_ISIN_TRANSITIONS.csv
13_MUHURAT_SESSION_AUDIT.csv
17_MEMBER_GAP_EPISODES.csv
19_KEY_DATASET_MANIFEST.csv
20_PYTHON_SCRIPT_MANIFEST.csv
21_CRITICAL_FAILURES.csv
22_REVIEW_ITEMS.csv
23_OVERALL_STATUS.csv

INTERPRETATION
--------------
PASS:
    Data-quality closure is complete.

PASS_WITH_REVIEW_ITEMS:
    Core integrity checks passed. Review the small number of documentary /
    identity items before final memo wording.

FAIL_REVIEW_REQUIRED:
    A serious integrity inconsistency exists, but all diagnostics have been
    saved so the exact cause can be fixed without rerunning blind.

IMPORTANT
---------
This audit does not define investability filters, F&O eligibility, pair
selection, SSD, Engle-Granger, trading rules, costs or OOS results.
"""

(
    OUTPUT_DIR
    /
    "README_DATA_QUALITY_CLOSURE_V2.txt"
).write_text(
    readme,
    encoding="utf-8"
)


# =============================================================================
# 16. CONSOLE REPORT
# =============================================================================

print("\n")
print("=" * 110)
print("AUDIT FINISHED SUCCESSFULLY")
print("=" * 110)

print(
    f"\nOVERALL STATUS: {overall_status}"
)

print(
    f"Critical failures: {len(critical_failures)}"
)

print(
    f"Review items: {len(review_items)}"
)

print("\nSUMMARY")
print("-" * 110)

print(
    closure_summary.to_string(
        index=False
    )
)


if not critical_failures.empty:

    print("\nCRITICAL FAILURES — program did NOT abort")
    print("-" * 110)

    print(
        critical_failures.to_string(
            index=False
        )
    )


if not review_items.empty:

    print("\nREVIEW ITEMS")
    print("-" * 110)

    print(
        review_items.to_string(
            index=False
        )
    )


print("\nOutputs saved to:")
print(
    OUTPUT_DIR
)

print("\nUpload these files:")
for filename in [
    "00_DATA_QUALITY_CLOSURE_SUMMARY.csv",
    "07_ISIN_TRANSITIONS.csv",
    "08_UNEXPLAINED_ISIN_TRANSITIONS.csv",
    "13_MUHURAT_SESSION_AUDIT.csv",
    "17_MEMBER_GAP_EPISODES.csv",
    "19_KEY_DATASET_MANIFEST.csv",
    "20_PYTHON_SCRIPT_MANIFEST.csv",
    "21_CRITICAL_FAILURES.csv",
    "22_REVIEW_ITEMS.csv",
    "23_OVERALL_STATUS.csv",
]:

    print(
        OUTPUT_DIR
        /
        filename
    )

print("\n")
print("=" * 110)
print(
    "NO PRICE, RETURN, VOLUME, MEMBERSHIP OR STRATEGY DATA WAS MODIFIED."
)
print("=" * 110)
